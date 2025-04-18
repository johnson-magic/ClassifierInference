#include "utils.h"

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect) {

    cv::Point2f vertices[4];
	


    rotatedRect.points(vertices);

   
    for(int i = 0; i < 4; ++i) {
		cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
	}
      
        
}



void printRotatedRect(const cv::RotatedRect& rotatedRect) {
    cv::Point2f center = rotatedRect.center; 
    cv::Size2f size = rotatedRect.size;       
    float angle = rotatedRect.angle;         

    std::cout << "RotatedRect:" << std::endl;
    std::cout << "Center: (" << center.x << ", " << center.y << ")" << std::endl;
    std::cout << "Size: (" << size.width << ", " << size.height << ")" << std::endl;
    std::cout << "Angle: " << angle << " degrees" << std::endl;
}


bool hasImageUpdated(const std::string& image_path, std::filesystem::file_time_type& lastCheckedTime) {
    
    if (!std::filesystem::exists(image_path)) {
        std::cout << "file does not exists: " << image_path << std::endl;
        return false;
    }


    std::filesystem::file_time_type curWriteTime = std::filesystem::last_write_time(image_path);
    
    if (curWriteTime != lastCheckedTime) {
        lastCheckedTime = curWriteTime;
        return true;
    }
    
    return false;
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

// 分离文件名和扩展名的函数
std::pair<std::string, std::string> splitext(const std::string& filename) {
    size_t lastDot = filename.find_last_of(".");
    if (lastDot == std::string::npos) {
        // 找不到扩展名，返回原文件名和空字符串
        return {filename, ""};
    } else {
        // 分离文件名和扩展名
        return {filename.substr(0, lastDot), filename.substr(lastDot)};
    }
}

void SaveRotatedObjsToTextFile(std::vector<RotatedObj>& rotated_objs, const std::string& filename){

    auto [name, ext] = splitext(filename);

    for(auto rotated_obj : rotated_objs){
        std::string cur_filename = name + "_" + std::to_string(rotated_obj.class_index) + ".txt";

        // 打开文件
        std::ofstream outFile(cur_filename);
        if (!outFile) {
            std::cerr << "无法打开文件 " << filename << std::endl;
            return;
        }

        outFile << rotated_obj.class_index << std::endl;
        outFile << rotated_obj.score << std::endl;

        cv::Point2f vertices[4];
        rotated_obj.rotated_rect.points(vertices);
        for(auto vertice : vertices){
            outFile << vertice.x << std::endl;
            outFile << vertice.y << std::endl;
        }

        // 关闭文件
        outFile.close();
    }




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
 
