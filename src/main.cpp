//#define _CRT_NO_VA_START_VALIDATION
#include "main.h"

int main(int argc, char* argv[]){

	if (argc < 7 || argc > 9) {
		std::cerr << " ERROR main.cpp: Not enough params = " << argc << std::endl;
		exit(1);
	}

	std::string prefix = "";
	std::string nameCurrFrame = "";
	std::string nameNextFrame = "";
	std::string dirRes = "";
	std::string dirParam = "";
	std::string nameOptParamFile = "";
	std::string nameInitFlow_F = "";
	std::string nameInitFlow_B = "";

	if (argc == 7){
		prefix = argv[1];
		nameCurrFrame = argv[2];
		nameNextFrame = argv[3];
		dirRes = argv[4];
		dirParam = argv[5];
		nameOptParamFile = argv[6];
	}
	else if (argc == 8){
		prefix = argv[1];
		nameCurrFrame = argv[2];
		nameNextFrame = argv[3];
		dirRes = argv[4];
		dirParam = argv[5];
		nameOptParamFile = argv[6];
		nameInitFlow_F = argv[7];
	}
	else if (argc == 9){
		prefix = argv[1];
		nameCurrFrame = argv[2];
		nameNextFrame = argv[3];
		dirRes = argv[4];
		dirParam = argv[5];
		nameOptParamFile = argv[6];
		nameInitFlow_F = argv[7];
		nameInitFlow_B = argv[8];
	}

	const std::string nameFlowResult_F = ((std::string)dirRes).append(prefix).append("_10.png");
	const std::string nameFlowResult_B = ((std::string)dirRes).append("flow_b_").append(prefix).append(".png");
	const std::string nameOccResult_F = ((std::string)dirRes).append("occ_").append(prefix).append("_f.png");
	const std::string nameOccResult_B = ((std::string)dirRes).append("occ_").append(prefix).append("_b.png");

	// MirrorFlow processing
	MirrorFlow mf;
	mf.process(nameCurrFrame, nameNextFrame,
		prefix,
		dirParam,
		nameOptParamFile,
		nameInitFlow_F, nameInitFlow_B,
		dirRes,
		nameFlowResult_F, nameFlowResult_B,
		nameOccResult_F, nameOccResult_B);
		
	return 1;
}