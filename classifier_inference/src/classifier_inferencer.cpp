#include "classifier_inferencer.h"

void ClassifierInferencer::Init(const std::string &model_path){
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "default");
    
    Ort::SessionOptions option;
    option.SetIntraOpNumThreads(1);
    option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_ = new Ort::Session(env, ConvertToWString(model_path).c_str(), option);
}

void ClassifierInferencer::GetInputInfo(){
	numInputNodes_ = GetSessionInputCount();
    for (int input_id = 0; input_id < numInputNodes_; input_id++) {
        Ort::AllocatorWithDefaultOptions allocator;  // 如何理解allocator的工作机制？ 它能够被复用吗？
        auto input_name = session_->GetInputNameAllocated(input_id, allocator);  // 通常，numInputNodes_只为1
        input_node_names_.push_back(input_name.get());  // char* -> string
	
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(input_id);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        net_h_.push_back(input_dims[2]);
        net_w_.push_back(input_dims[3]);    
	}
}

size_t ClassifierInferencer::GetSessionInputCount(){
    return session_->GetInputCount();
}

size_t ClassifierInferencer::GetSessionOutputCount(){
    return session_->GetOutputCount();
}

void ClassifierInferencer::GetOutputInfo(){
	numOutputNodes_ = GetSessionOutputCount();
    
    for(int output_id = 0; output_id < numOutputNodes_; output_id++){
        Ort::AllocatorWithDefaultOptions allocator;
        auto out_name = session_->GetOutputNameAllocated(output_id, allocator);
        output_node_names_.push_back(out_name.get());

        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(output_id);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        // output_h_.push_back(output_dims[1]);  // 注意：对于检测模型，模型的输出一般为(B, 8, anchor_point)
        // output_w_.push_back(output_dims[2]);
        class_num_.push_back(output_dims[1]);   
    } 
}

void ClassifierInferencer::PreProcess(const std::string& image_path){
    image_path_ = image_path;
    std::ifstream test_open(image_path_);
    if(test_open.is_open()){
        image_ = cv::imread(image_path_);
        if (image_.empty()) {
	    	std::cerr << "Failed to read the image!" << std::endl;
            return;
        }
        test_open.close(); 
    }else{
        std::cerr << "Failed to read the image!" << std::endl;
        return;
    }
	image_ = pad_and_resize(image_);
}

void ClassifierInferencer::PreProcess(cv::Mat image){
	image_ = image;
	if (image_.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		return;
	}
	image_ = pad_and_resize(image_);
}

void ClassifierInferencer::SaveOrtValueAsImage(Ort::Value& value, const std::string& filename) {
    // 确保值是张量
    if (!value.IsTensor()) {
        std::cerr << "Value is not a tensor." << std::endl;
        return;
    }

    Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
    
    // 获取张量的维度
    std::vector<int64_t> shape = info.GetShape();
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);
    
    // 检查是否为 RGB 图像，形状应为 {1, 3, height, width}
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        std::cerr << "Expected a 4D tensor with shape {1, 3, height, width}." << std::endl;
        return;
    }

    // 获取张量数据
    float* data = value.GetTensorMutableData<float>();

    // 将数据转为 OpenCV 的 cv::Mat 格式，注意通道顺序
    cv::Mat image(height, width, CV_32FC3, data);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR); // 转换为 BGR 格式以便保存

    // 将数据类型转换为可保存的格式（如 8 位无符号整数）
    cv::Mat imageToSave;
    image.convertTo(imageToSave, CV_8UC3, 255.0); // 假设输入是范围在 [0, 1] 之间的浮点数

    // 保存图像
    if (!cv::imwrite(filename, imageToSave)) {
        std::cerr << "Failed to save image to " << filename << std::endl;
	}
}

void ClassifierInferencer::Inference(){
	const std::array<const char*, 1> inputNames = { input_node_names_[0].c_str() };  // std::array用于fixed size array
	const std::array<const char*, 1> outNames = { output_node_names_[0].c_str() };
   
    cv::Mat blob;
	cv::dnn::blobFromImage(image_, blob, 1 / 255.0, cv::Size(net_w_[0], net_h_[0]), cv::Scalar(0, 0, 0), true, false);  // swapRB = true, crop = false,
    
	size_t tpixels = net_h_[0] * net_w_[0] * 3 * 1;
	std::array<int64_t, 4> input_shape_info{ 1, 3, net_h_[0], net_w_[0]};

    
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    assert(input_tensor.IsTensor());
    
	try {
		#ifdef CONFORMANCE_TEST
			SaveOrtValueToTextFile(input_tensor, "onnx_input.txt");
		#endif
		ort_outputs_ = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor, inputNames.size(), outNames.data(), outNames.size());
		#ifdef CONFORMANCE_TEST
			for(int i=0; i<ort_outputs_.size(); i++){
				SaveOrtValueToTextFile(input_tensor, "onnx_output_" + std::to_string(i) + ".txt");
			}
		#endif
    }
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
}

void ClassifierInferencer::PostProcess(){
    predictions_.clear();
    scores_.clear();
	const float* pdata = ort_outputs_[0].GetTensorMutableData<float>();
	int prediction = 0;
    float max_prob = *(pdata + prediction);
    int begin_id = 1;
    for(int i = begin_id; i<class_num_[0]; i++){
        if(*(pdata + i) > max_prob){
            max_prob = *(pdata + i);
            prediction = i;
        }
    }
    predictions_.push_back(prediction);
    scores_.push_back(max_prob);
	#ifdef CONFORMANCE_TEST
		SaveRotatedObjsToTextFile(remain_rotated_objects_, "remain_rotated_objects.txt");
	#endif

}

cv::Mat ClassifierInferencer::pad_and_resize(const cv::Mat &img){
	/*如果原始图片的宽和高有任意1个超过网络的输入宽、高，则先按照长宽比进行padding，再resize；
      否则，直接padding到网络的输入宽、高*/
    
    int w_img = img.cols;
    int h_img = img.rows;
    
    // 计算缩放比例
    float scale = std::min(static_cast<double>(net_h_[0]) / h_img, static_cast<double>(net_w_[0]) / w_img);
    scale = std::min(static_cast<double>(scale), 1.0);  //如果原始图片的宽和高都不超过网络的宽和高，则将scale设置为1，表示事实上不进行resize.
    // 计算新的宽度和高度
    int w_resized = static_cast<int>(std::round(scale * w_img));
    int h_resized = static_cast<int>(std::round(scale * h_img));
    // 缩放图像
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(w_resized, h_resized), 0, 0, cv::INTER_LINEAR);

    // 计算边框宽度和高度
    int dw = net_w_[0] - w_resized;
    int dh = net_h_[0] - h_resized;

    // 分配边框宽度（上下左右）
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    // 添加边框
    cv::Mat img_bordered;
    cv::copyMakeBorder(img_resized, img_bordered, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 返回处理后的图像
    return img_bordered;
}

std::pair<std::vector<int>, std::vector<float>> ClassifierInferencer::GetRes(){
    return std::make_pair(predictions_, scores_);
}

void ClassifierInferencer::Release(){
	session_->release();
}


