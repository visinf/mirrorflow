#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "graph_superpixel.h"	// State
#include "superpixels.h"		// Superpixel
//#include "initialflow.h"		// InitialFlow
#include "OptModule.h"	// Optimization
#include <time.h>

class MirrorFlow {

public:

	MirrorFlow() {
		segments_F_.resize(0);
		boundaries_F_.resize(0);
		segments_B_.resize(0);
		boundaries_B_.resize(0);
		
	}

	struct MirrorFlowParameters {

		// verbose
		bool verbose_main = false;
		bool verbose_seg = false;

		// main algorithm
		float bWeight = 2.0f;
	};

	void process(const std::string& nameCurrFrame, 
		const std::string& nameNextFrame,
		const std::string& prefix, 
		const std::string& dirParam,
		const std::string& nameOptParamFile,
		const std::string& nameInitFlow_F,
		const std::string& nameInitFlow_B,
		const std::string& dirRes,
		const std::string& nameFlowResult_F, const std::string& nameFlowResult_B,
		const std::string& nameOccResult_F,	const std::string& nameOccResult_B);


private:

	// MirrorFlow
	MirrorFlowParameters readParamsMF(const std::string& fileName);
	void readParamsOpt(const std::string& fileName, float& bPixelWeight, float& superpixelWeight);
	void calculateBoundaryWeights(const std::string nameImage, std::vector<Segment>& segments, std::vector<Boundary>& boundaries);
	void ReadFlowField(const std::string file_name, cv::Mat& flowX, cv::Mat& flowY);

	// params
	const std::string fileNameParams_ = "params_mirrorFlow.yml";

	// State - Superpixel graph
	std::vector<Segment> segments_F_;
	std::vector<Boundary> boundaries_F_;
	std::vector<Segment> segments_B_;
	std::vector<Boundary> boundaries_B_;
	cv::Mat superpixelLabelMap_F_;
	cv::Mat superpixelLabelMap_B_;

	// Initial Flow
	cv::Mat initFlowX_F_;
	cv::Mat initFlowY_F_;
	cv::Mat initFlowX_B_;
	cv::Mat initFlowY_B_;

	// image
	int width_;
	int height_;
};