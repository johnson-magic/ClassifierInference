#include <windows.h>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "time_limit.h"

bool hasImageUpdated(const std::string& image_path, cv::Scalar &pre_pixel_sum);
long long GetSecondsInterval(SYSTEMTIME start, SYSTEMTIME end);
void SaveOrtValueToTextFile(Ort::Value& ortValue, const std::string& filename);
void readFromBinaryFile(const std::string& filename, const TimeLimit& timelimit);
void saveToBinaryFile(const TimeLimit& timelimit, const std::string& filename);
int encrypt(int number, int key);
int decrypt(int encrypted_number, int key);
