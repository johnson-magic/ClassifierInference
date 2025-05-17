#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>

#include "utils.h"
#include "classifier_inferencer.h"

using namespace std;

volatile bool keepRunning = true;

BOOL WINAPI HandleCtrlC(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        keepRunning = false;
    }
    return TRUE;
}

int main(int argc, char** argv){
	#ifdef ENCRYPT
		TimeLimit timelimit;
		readFromBinaryFile("onnx.dll", timelimit);
		int left = decrypt(timelimit.left, 20250124);
	#endif

	if(argc != 4){  // 分类任务，没有什么可可视化的，因此参数为4，而非5
		std::cout<<"[ERROR] classifier_inference model_path img_path result_path"<<std::endl;
		std::cout<<"e.g., ./classifier_inference.exe text_direction_classify.onnx ./data/test.bmp res.txt"<<std::endl;
		return 0;
	}

	std::string model_path = argv[1];
	std::string image_path = argv[2];
	std::string result_path = argv[3];
    
	ClassifierInferencer classifier(model_path);
    classifier.GetInputInfo();
	classifier.GetOutputInfo();

	cv::Scalar pre_pixel_sum=cv::Scalar(0, 0, 0, 0);
    while (keepRunning) {
        if (hasImageUpdated(image_path, pre_pixel_sum)) {

			#ifdef ENCRYPT
				if(left == 0){
					std::cerr<<"Error 3, please contact the author!"<<std::endl;
					return 0;
				}
				left = left - 1;
				timelimit.left = encrypt(left, 20250124);
				saveToBinaryFile(timelimit, "onnx.dll");
			#endif

			int iter = 1;

				#ifdef SPEED_TEST
					iter = 5;
					SYSTEMTIME start, end_preprocess, end_inferencer, end_postprocess, end_process, end_saveres, end_visres;
					GetSystemTime(&start);
					std::cout<<"**************************************GetSystemTime(&start)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				classifier.PreProcess(image_path);
			}
		
				#ifdef SPEED_TEST
					GetSystemTime(&end_preprocess);
					std::cout<<"**************************************GetSystemTime(&end_preprocess)****************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				classifier.Inference();
			}
				
				#ifdef SPEED_TEST
					GetSystemTime(&end_inferencer);
					std::cout<<"**************************************GetSystemTime(&end_inferencer)****************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				classifier.PostProcess();
			}

				#ifdef SPEED_TEST
					GetSystemTime(&end_postprocess);
					std::cout<<"**************************************GetSystemTime(&end_postprocess)***************************"<<std::endl;
				#endif
            
            for(int i=0; i< iter; i++){
            	std::pair<std::vector<int>, std::vector<float>> res = classifier.GetRes();
                std::vector<int> classes = res.first;
                std::vector<float> scores = res.second;
                for(int j=0; j<classes.size(); j++){
                    std::cout<<"class: "<<classes[j]<<"scores: "<<scores[j]<<std::endl;
                }
			}

				#ifdef SPEED_TEST
					GetSystemTime(&end_saveres);
					std::cout<<"**************************************GetSystemTime(&end_saveres)*******************************"<<std::endl;
				#endif


				#ifdef SPEED_TEST
					std::cout<<"total timecost: "<< (GetSecondsInterval(start, end_postprocess))/iter<<"ms"<<std::endl;
				    std::cout<<"preprocess of inferencer timecost: "<<(GetSecondsInterval(start, end_preprocess))/iter<<"ms"<<std::endl;
					std::cout<<"inference of inferencer timecost: "<<(GetSecondsInterval(end_preprocess, end_inferencer))/iter<<"ms"<<std::endl;
					std::cout<<"postprocess of inferencer timecost: "<<(GetSecondsInterval(end_inferencer, end_postprocess))/iter<<"ms"<<std::endl;
                    std::cout<<"save result in txt of angle detector timecost: "<<(GetSecondsInterval(end_process, end_saveres))/iter<<"ms"<<std::endl;
				#endif
			std::cout << "finished, waiting ..." << std::endl;
        }
    }

	classifier.Release();  // session_options.release(); is it ok?

	std::cout << "exit after 1 minutes" << std::endl;
	std::this_thread::sleep_for(std::chrono::minutes(1));
    return 0;
}
