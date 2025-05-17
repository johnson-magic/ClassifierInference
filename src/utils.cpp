#include "utils.h"

bool hasImageUpdated(const std::string& img_path, cv::Scalar &pre_pixel_sum) {
    
    std::ifstream test_open(img_path);
    if(test_open.is_open()){
        cv::Mat img_temp = cv::imread(img_path);
        test_open.close();
        if (img_temp.empty()) {
	    	std::cerr << "Failed to read the image!" << std::endl;
            return false;
        }
        cv::Scalar cur_pixel_sum=cv::sum(img_temp);
        if(pre_pixel_sum == cur_pixel_sum){
            return false;
        }
        else{
            pre_pixel_sum = cur_pixel_sum;
            return true;
        }
    }
    else{
        return false;
    }
}

long long GetSecondsInterval(SYSTEMTIME start, SYSTEMTIME end) {
    FILETIME ftStart, ftEnd;
    ULARGE_INTEGER ullStart, ullEnd;

    // 将 SYSTEMTIME 转换为 FILETIME
    SystemTimeToFileTime(&start, &ftStart);
    SystemTimeToFileTime(&end, &ftEnd);
    
    // 将 FILETIME 转换为 ULARGE_INTEGER 以便进行算术运算
    ullStart.u.LowPart = ftStart.dwLowDateTime;
    ullStart.u.HighPart = ftStart.dwHighDateTime;
    ullEnd.u.LowPart = ftEnd.dwLowDateTime;
    ullEnd.u.HighPart = ftEnd.dwHighDateTime;

    // 计算时间间隔（单位是 100 纳秒）
    long long interval = ullEnd.QuadPart - ullStart.QuadPart;

    // 将间隔转换为毫秒（1毫秒 = 10,000 100 纳秒）
    return interval / 10000;
}

void SaveOrtValueToTextFile(Ort::Value& ortValue, const std::string& filename) {
    // 获取数据类型
    Ort::TensorTypeAndShapeInfo  tensor_type_and_shape_info = ortValue.GetTensorTypeAndShapeInfo();
    auto elementType = tensor_type_and_shape_info.GetElementType();

    // 获取数据指针和数据大小
    void* data = ortValue.GetTensorMutableData<void>();
    size_t dataSize = ortValue.GetTensorTypeAndShapeInfo().GetElementCount();

    // 打开文件
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    // 将数据写入文件
    if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float* floatData = static_cast<float*>(data);
        for (size_t i = 0; i < dataSize; ++i) {
            outFile << floatData[i] << std::endl;
        }
    } else if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        int32_t* intData = static_cast<int32_t*>(data);
        for (size_t i = 0; i < dataSize; ++i) {
            outFile << intData[i] << std::endl;
        }
    }
    // 你可以根据需要添加其他类型的处理...

    // 关闭文件
    outFile.close();
}

void readFromBinaryFile(const std::string& filename, const TimeLimit& timelimit) {
    // 读取二进制文件
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr <<"Error 1, please contact the author!"<< filename << std::endl;
        return;
    }

    infile.read((char*)&timelimit, sizeof(timelimit));
    infile.close();
}


void saveToBinaryFile(const TimeLimit& timelimit, const std::string& filename) {
    // 打开二进制文件用于写入
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error 2, please contact the author!" << filename << std::endl;
        return;
    }

    outfile.write((char*)&timelimit, sizeof(timelimit));
    outfile.close();
}

#include <iostream>
 
int encrypt(int number, int key) {
    return number ^ key;
}
 
int decrypt(int encrypted_number, int key) {
    return encrypted_number ^ key;
}
 
