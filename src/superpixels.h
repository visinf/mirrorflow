#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "graph_superpixel.h"

class Superpixels{

public:

	Superpixels(float bPixelWeight = 0.015f, float superpixelWeight = 0.02f){
		bPixelWeight_ = bPixelWeight;
		superpixelWeight_ = superpixelWeight;
	}

	// tpseg
	bool processTPSeg(const std::string& nameCurrFrame, std::vector<Segment>& segments, std::vector<Boundary>& boundaries, cv::Mat& superpixelLabelMap, const std::string& paramDir, bool verbose);


private:

	// tpseg
	void saveSegmentAndBoundaryFromTPSeg(cv::Mat& image, std::vector<Segment>& segments, std::vector<Boundary>& boundaries, cv::Mat& labelInfo);
	void saveNNeighborIndex(std::vector<Segment>& segments);

	// general
	void ShowResizeImage(const std::string& windowName, const cv::Mat& srcImage, int xx = 500, int yy = 500);
	bool checkSuperpixelValidity(cv::Mat& labelInfo);

	// graph-superpixel
	void updateBoundary(cv::Mat& image, int xx1, int yy1, int segment1, int xx2, int yy2, int segment2, std::vector<Segment>& segments, std::vector<Boundary>& boundaries);
	int getBoundaryIndex(int segment1, int segment2, std::vector<Boundary>& boundaries);
	void drawBoundaryAndSegementResult(cv::Mat& image, std::vector<Segment>& segments, std::vector<Boundary>& boundaries);
	void calculateBoundaryWeights(const cv::Mat& image, std::vector<Segment>& segments, std::vector<Boundary>& boundaries);

	// general
	int height_;
	int width_;
	
	// tpseg
	const std::string fileNameParamsTPSeg_ = "params_tpseg.yml";

	// boundary weight
	float bPixelWeight_;
	float superpixelWeight_;

};


