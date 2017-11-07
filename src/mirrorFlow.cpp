#include "mirrorFlow.h"


void MirrorFlow::process(const std::string& nameCurrFrame,
	const std::string& nameNextFrame,
	const std::string& prefix,
	const std::string& dirParam,
	const std::string& nameOptParamFile,
	const std::string& nameInitFlow_F,
	const std::string& nameInitFlow_B,
	const std::string& dirRes,
	const std::string& nameFlowResult_F, const std::string& nameFlowResult_B,
	const std::string& nameOccResult_F, const std::string& nameOccResult_B){

	// Initial configuration
	MirrorFlow::MirrorFlowParameters params;
	params = readParamsMF(((std::string)dirParam).append("/").append(fileNameParams_));
	cv::Mat currFrame = cv::imread(nameCurrFrame, cv::IMREAD_GRAYSCALE);
	cv::Mat nextFrame = cv::imread(nameNextFrame, cv::IMREAD_GRAYSCALE);
		
	// Opt parameter file define
	std::string fullPathOptParamFile = (std::string(dirParam)).append("/").append(nameOptParamFile);
	float bPixelWeight = 0.f;
	float superpixelWeight = 0.f; 
	readParamsOpt(fullPathOptParamFile, bPixelWeight, superpixelWeight);	

	// load image
	if (currFrame.empty()) {
		std::cout << " ERROR mirrorFlow.cpp : Cannot find the image = " << nameCurrFrame << std::endl;
		return;
	}
	if (nextFrame.empty()) {
		std::cout << " ERROR mirrorFlow.cpp : Cannot find the image = " << nameNextFrame << std::endl;
		return;
	}
	if (currFrame.cols != nextFrame.cols) {
		std::cout << " ERROR mirrorFlow.cpp : input image dimension mismatch = " << currFrame.cols << " != " << nextFrame.cols << std::endl;
		return;
	}
	if (currFrame.rows != nextFrame.rows) {
		std::cout << " ERROR mirrorFlow.cpp : input image dimension mismatch = " << currFrame.rows << " != " << nextFrame.rows << std::endl;
		return;
	}
	width_ = currFrame.cols;
	height_ = currFrame.rows;

	// Preprocessing - 1) superpixel initialization
	if (params.verbose_main) {
		std::cout << " Superpixel Initialization.." << std::endl;
	}
	Superpixels spixel(bPixelWeight, superpixelWeight);
	if (!spixel.processTPSeg(nameCurrFrame, segments_F_, boundaries_F_, superpixelLabelMap_F_, dirParam, params.verbose_seg)) {
		return;
	}
	if (!spixel.processTPSeg(nameNextFrame, segments_B_, boundaries_B_, superpixelLabelMap_B_, dirParam, params.verbose_seg)) {
		return;
	}
	
	// Preprocessing - 2) initial optical flow
	if (params.verbose_main) {
		std::cout << " Initial Flow Estimation.." << std::endl;
	}
	
	if (nameInitFlow_F.length() == 0){
		std::cout << " ERROR mirrorFlow.cpp : no Input Flow _ forward"<< std::endl;
		return;
	}
	else{
		ReadFlowField(nameInitFlow_F, initFlowX_F_, initFlowY_F_);
	}
	
	if (nameInitFlow_B.length() == 0) {
		std::cout << " ERROR mirrorFlow.cpp : no Input Flow _ backward" << std::endl;
		return;
	}
	else {		
		ReadFlowField(nameInitFlow_B, initFlowX_B_, initFlowY_B_);
	}


	// Optimization
	if (params.verbose_main) {
		std::cout << " Optimization..!! " << std::endl;
	}	
	
	OptModule* optimizer = 0;
	optimizer = new OptModule(fullPathOptParamFile);
	optimizer->initializeVariables(
		nameCurrFrame, nameNextFrame,
		initFlowX_F_, initFlowX_B_,
		initFlowY_F_, initFlowY_B_,
		segments_F_, segments_B_, 
		boundaries_F_, boundaries_B_,
		superpixelLabelMap_F_, superpixelLabelMap_B_,
		dirRes, prefix);

	optimizer->solve_QPBO_LAE();	
	
	// save results
	optimizer->WriteFlowField(nameFlowResult_F, nameFlowResult_B, nameOccResult_F, nameOccResult_B);
	
	optimizer->FreeMemory();

	if (params.verbose_main) {
		std::cout << " Done..!" << std::endl;		
	}
}

MirrorFlow::MirrorFlowParameters MirrorFlow::readParamsMF(const std::string& fileName) {

	MirrorFlow::MirrorFlowParameters params;

	cv::FileStorage fs(fileName, cv::FileStorage::READ);

	if (!fs.isOpened()) {
		std::cout << " WARNING mirrorFlow.cpp : Cannot find the parameter file = " << fileName << std::endl;
		return params;
	}

	if (!fs["verbose_main"].empty()) {
		fs["verbose_main"] >> params.verbose_main;
	}
	if (!fs["verbose_seg"].empty()) {
		fs["verbose_seg"] >> params.verbose_seg;
	}

	fs.release();

	return params;
}

void MirrorFlow::readParamsOpt(const std::string& fileName, float& bPixelWeight, float& superpixelWeight) {

	cv::FileStorage fs(fileName, cv::FileStorage::READ);

	if (!fs.isOpened()) {
		std::cout << " WARNING mirrorFlow.cpp : Cannot find the parameter file = " << fileName << std::endl;
		return;
	}

	if (!fs["bPixelWeight"].empty()) {
		fs["bPixelWeight"] >> bPixelWeight;
	}
	if (!fs["superpixelWeight"].empty()) {
		fs["superpixelWeight"] >> superpixelWeight;
	}

	fs.release();	
}

void MirrorFlow::ReadFlowField(const std::string file_name, cv::Mat& flowX, cv::Mat& flowY){

	png::image< png::rgb_pixel_16 > image(file_name);
	int width__ = image.get_width();
	int height__ = image.get_height();
	if (height__ != height_ || width__ != width_){
		std::cout << "Fatal error < flow size is not the same " << std::endl;
	}
	flowX = cv::Mat(height__, width__, CV_32FC1);
	flowY = cv::Mat(height__, width__, CV_32FC1);

	for (int32_t v = 0; v < height__; v++) {
		for (int32_t u = 0; u < width__; u++) {
			png::rgb_pixel_16 val = image.get_pixel(u, v);
			if (val.blue>0) {
				flowX.at<float>(v, u) = ((float)val.red - 32768.0f) / 64.0f;
				flowY.at<float>(v, u) = ((float)val.green - 32768.0f) / 64.0f;
			}
		}
	}
}

