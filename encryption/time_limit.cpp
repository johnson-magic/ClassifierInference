
#include "time_limit.h"

void saveToBinaryFile(const TimeLimit& timelimit, const std::string& filename) {
    // 打开二进制文件用于写入
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    

    outfile.write((char*)&timelimit, sizeof(timelimit));
    outfile.close();
}

int main(int argc, char** argv){
    if(argc != 2){
		std::cout<<"[ERROR] timelimit time"<<std::endl;
		std::cout<<"e.g., ./timelimit.exe 1000"<<std::endl;
		return 0;
	}

    TimeLimit timelimit;
    timelimit.name = "onnx";
    timelimit.left = encrypt(std::stoi(argv[1]), 20250124);
    std::cout<<timelimit.left<<std::endl;
    saveToBinaryFile(timelimit, "onnx.dll");


	

}