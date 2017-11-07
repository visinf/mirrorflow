#pragma once
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>  
#include "superpixels.h"
#include "tpseg/stdafx.h"		//-------//
#include "tpseg/segengine.h"	//		 //
#include "tpseg/functions.h"	// tpseg //
#include "tpseg/utils.h"		//	 	 //
#include <fstream>				//	 	 //	
#include <cstdlib>				//-------//



bool Superpixels::checkSuperpixelValidity(cv::Mat& labelInfo){

	int maxLabel = -1;
	for (int hh = 0; hh < labelInfo.rows; ++hh){
		for (int ww = 0; ww < labelInfo.cols; ++ww){
			int currLabel = (int)labelInfo.at<unsigned short>(hh, ww);
			if (currLabel > maxLabel){
				maxLabel = currLabel;
			}
		}
	}

	std::vector<int> segments;
	segments.resize(maxLabel + 1, 0);

	for (int xx = 0; xx < width_; xx++){
		for (int yy = 0; yy < height_; yy++){

			int currLabel = (int)labelInfo.at<unsigned short>(yy, xx);

			// save the pixel in the segment
			segments[currLabel] = segments[currLabel] + 1;
		}
	}
	
	for (int ii = 0; ii < segments.size(); ++ii){
		if (segments[ii] < 6){
			return false;
		}
	}

	return true;
}


void Superpixels::saveSegmentAndBoundaryFromTPSeg(cv::Mat& image, std::vector<Segment>& segments, std::vector<Boundary>& boundaries, cv::Mat& labelInfo){

	int maxLabel = -1;

	for (int hh = 0; hh < labelInfo.rows; ++hh){
		for (int ww = 0; ww < labelInfo.cols; ++ww){
			int currLabel = (int)labelInfo.at<unsigned short>(hh, ww);
			if (currLabel > maxLabel){
				maxLabel = currLabel;
			}
		}
	}

	segments.resize(maxLabel+1);

	for (int xx = 0; xx < width_; xx++){
		for (int yy = 0; yy < height_; yy++){

			int currLabel = (int)labelInfo.at<unsigned short>(yy, xx);

			// save the pixel in the segment
			segments[currLabel].appendPixel(xx, yy);

			// check right pixel whether it is a neighboring pixel & save
			if (xx + 1 < width_){
				int rightLabel = (int)labelInfo.at<unsigned short>(yy, xx+1);
				if (currLabel != rightLabel){
					updateBoundary(image, xx, yy, currLabel, xx + 1, yy, rightLabel, segments, boundaries);
				}
			}

			// check below pixel whether it is a neighboring pixel & save
			if (yy + 1 < height_){
				int belowlabel = (int)labelInfo.at<unsigned short>(yy+1, xx);
				if (currLabel != belowlabel){
					updateBoundary(image, xx, yy, currLabel, xx, yy + 1, belowlabel, segments, boundaries);
				}
			}

			// check right below pixel whether it is a neighboring pixel & save
			if (xx + 1 < width_ && yy + 1 < height_){
				int rightBelowLabel = (int)labelInfo.at<unsigned short>(yy + 1, xx + 1);
				if (currLabel != rightBelowLabel){
					updateBoundary(image, xx, yy, currLabel, xx + 1, yy + 1, rightBelowLabel, segments, boundaries);
				}
			}

			// check left below pixel whether it is a neighboring pixel & save
			if (xx - 1 >= 0 && yy + 1 < height_){
				int leftBelowlabel = (int)labelInfo.at<unsigned short>(yy + 1, xx - 1);
				if (currLabel != leftBelowlabel){
					updateBoundary(image, xx, yy, currLabel, xx - 1, yy + 1, leftBelowlabel, segments, boundaries);
				}
			}
		}
	}

	// find the center point & top-left point of each segments	
	cv::Mat gray_mat;
	cv::cvtColor(image, gray_mat, CV_BGR2GRAY);

	for (int ss = 0; ss < segments.size(); ++ss){

		std::vector<cv::Point2f> pixels = segments[ss].getPixelVector();
		float xx = 0;
		float yy = 0;
		int min_x = width_;
		int min_y = height_;
		int max_x = 0;
		int max_y = 0;
		
		for (int pp = 0; pp < pixels.size(); ++pp){
			xx += pixels[pp].x;
			yy += pixels[pp].y;
			if ((int)pixels[pp].x < min_x){
				min_x = (int)pixels[pp].x;
			}
			if ((int)pixels[pp].y < min_y){
				min_y = (int)pixels[pp].y;
			}
			if ((int)pixels[pp].x > max_x){
				max_x = (int)pixels[pp].x;
			}
			if ((int)pixels[pp].y > max_y){
				max_y = (int)pixels[pp].y;
			}
		}

		cv::Point2f centerP = cv::Point2f(xx / (float)pixels.size(), yy / (float)pixels.size());
		segments[ss].setCenterPoint(centerP);
		
		// patch census
		segments[ss].setCensusPatchTL(cv::Point2i(min_x, min_y));
		segments[ss].setCensusPatchBR(cv::Point2i(max_x, max_y));
		
		// corner point vector
		std::vector<cv::Point2f> vecCornerPts;
		vecCornerPts.push_back(cv::Point2f((float)min_x, (float)min_y));	// top left
		vecCornerPts.push_back(cv::Point2f((float)max_x, (float)min_y));	// top right
		vecCornerPts.push_back(cv::Point2f((float)min_x, (float)max_y));	// bottom left
		vecCornerPts.push_back(cv::Point2f((float)max_x, (float)max_y));	// bottom right
		segments[ss].setVecCornerPts(vecCornerPts);

		// A bounding box for LK warping
		cv::Size bBoxSize = cv::Size((max_x - min_x) * 2, (max_y - min_y) * 2);
		cv::Point2i bBoxTopLeftPt = cv::Point2i((int)(centerP.x - bBoxSize.width / 2.f), (int)(centerP.y - bBoxSize.height / 2.f));
		cv::Point2i bBoxBottomRightPt = cv::Point2i((int)(centerP.x + bBoxSize.width / 2.f), (int)(centerP.y + bBoxSize.height / 2.f));

		bBoxTopLeftPt.x = max(0, bBoxTopLeftPt.x);
		bBoxTopLeftPt.y = max(0, bBoxTopLeftPt.y);
		bBoxBottomRightPt.x = min(bBoxBottomRightPt.x, width_);
		bBoxBottomRightPt.y = min(bBoxBottomRightPt.y, height_);
		bBoxSize.width = bBoxBottomRightPt.x - bBoxTopLeftPt.x;
		bBoxSize.height = bBoxBottomRightPt.y - bBoxTopLeftPt.y;

		segments[ss].setBboxTLpoint(bBoxTopLeftPt);
		segments[ss].setBboxBRpoint(bBoxBottomRightPt);
		segments[ss].setBboxSize(bBoxSize);
		
		// Extract features to track
		std::vector<cv::Point2f> features;
		cv::Mat roiImage = gray_mat(cv::Rect(bBoxTopLeftPt.x, bBoxTopLeftPt.y, bBoxSize.width, bBoxSize.height)).clone();
		cv::goodFeaturesToTrack(roiImage, features, 30, 0.001, 2.5); //calculate the features for use in next iteration
		segments[ss].setFeatureToTrack(features);
		segments[ss].setImagePatch(roiImage);
	}

}

void Superpixels::saveNNeighborIndex(std::vector<Segment>& segments){

	// neighbor 0 == itself
	for (int ii = 0; ii < segments.size(); ++ii){
		std::set<int> neighbor0;
		neighbor0.insert(ii);
		segments[ii].setNeighborSegmentIndexNhops(neighbor0, 0);
	}
		
	// neighbor 1
	for (int ii = 0; ii < segments.size(); ++ii){
		std::set<int> neighbor1;
		
		int neighborNum = segments[ii].getNumberOfNeighbor();
		for (int jj = 0; jj < neighborNum; ++jj){
			int idx = segments[ii].getNeighbor(jj);
			if (idx == ii) {
				continue;
			}
			neighbor1.insert(idx);
		}

		segments[ii].setNeighborSegmentIndexNhops(neighbor1, 1);
	}

	// neighbor 2
	for (int ii = 0; ii < segments.size(); ++ii){
		std::set<int> neighbor2;
		std::set<int> neighbor1 = segments[ii].getNeighborSegmentIndexNhops(1);
		
		for (auto IterPos = neighbor1.begin(); IterPos != neighbor1.end(); ++IterPos){
			int seed = *IterPos;
			neighbor2.insert(seed);
			int neighborNum = segments[seed].getNumberOfNeighbor();
			for (int jj = 0; jj < neighborNum; ++jj){
				int idx = segments[seed].getNeighbor(jj);
				if (ii == idx){
					continue;
				}
				neighbor2.insert(idx);
			}
		}
		segments[ii].setNeighborSegmentIndexNhops(neighbor2, 2);
	}

	// neighbor 3
	for (int ii = 0; ii < segments.size(); ++ii){
		std::set<int> neighbor3;
		std::set<int> neighbor2 = segments[ii].getNeighborSegmentIndexNhops(2);

		for (auto IterPos = neighbor2.begin(); IterPos != neighbor2.end(); ++IterPos){
			int seed = *IterPos;
			neighbor3.insert(seed);

			int neighborNum = segments[seed].getNumberOfNeighbor();
			for (int jj = 0; jj < neighborNum; ++jj){
				int idx = segments[seed].getNeighbor(jj);
				if (idx == ii){
					continue;
				}
				neighbor3.insert(idx);
			}
		}
		segments[ii].setNeighborSegmentIndexNhops(neighbor3, 3);
	}

	// neighbor 4
	for (int ii = 0; ii < segments.size(); ++ii) {
		std::set<int> neighbor4;
		std::set<int> neighbor3 = segments[ii].getNeighborSegmentIndexNhops(3);

		for (auto IterPos = neighbor3.begin(); IterPos != neighbor3.end(); ++IterPos) {
			int seed = *IterPos;
			neighbor4.insert(seed);

			int neighborNum = segments[seed].getNumberOfNeighbor();
			for (int jj = 0; jj < neighborNum; ++jj) {
				int idx = segments[seed].getNeighbor(jj);
				if (idx == ii) {
					continue;
				}
				neighbor4.insert(idx);
			}
		}
		segments[ii].setNeighborSegmentIndexNhops(neighbor4, 4);
	}

	//// print
	//for (int ii = 0; ii < segments_.size(); ++ii){
	//	std::set<int> neighbor0 = segments_[ii].getNeighborSegmentIndexN(0);
	//	std::set<int> neighbor1 = segments_[ii].getNeighborSegmentIndexN(1);
	//	std::set<int> neighbor2 = segments_[ii].getNeighborSegmentIndexN(2);
	//	std::set<int> neighbor3 = segments_[ii].getNeighborSegmentIndexN(3);
	//	
	//	std::cout << "[" << ii << "]" << std::endl;
	//	std::cout << "Neighbor0: ";
	//	for (auto IterPos = neighbor0.begin(); IterPos != neighbor0.end(); ++IterPos){
	//		std::cout << *IterPos << " ";
	//	}
	//	std::cout << std::endl;
	//	std::cout << "Neighbor1: ";
	//	for (auto IterPos = neighbor1.begin(); IterPos != neighbor1.end(); ++IterPos){
	//		std::cout << *IterPos << " ";
	//	}
	//	std::cout << std::endl;
	//	std::cout << "Neighbor2: ";
	//	for (auto IterPos = neighbor2.begin(); IterPos != neighbor2.end(); ++IterPos){
	//		std::cout << *IterPos << " ";
	//	}
	//	std::cout << std::endl;
	//	std::cout << "Neighbor3: ";
	//	for (auto IterPos = neighbor3.begin(); IterPos != neighbor3.end(); ++IterPos){
	//		std::cout << *IterPos << " ";
	//	}
	//	std::cout << std::endl;
	//	std::cout << std::endl;
	//}	

}

void Superpixels::updateBoundary(cv::Mat& image, int xx1, int yy1, int segment_idx1, int xx2, int yy2, int segment_idx2, std::vector<Segment>& segments, std::vector<Boundary>& boundaries){

	// get boundary index & make a new one if doesn't exist before
	int boundaryIndex;
	boundaryIndex = getBoundaryIndex(segment_idx1, segment_idx2, boundaries);

	// save the point in the boundary
	boundaries[boundaryIndex].appendPoint((float)xx1, (float)yy1, (float)xx2, (float)yy2);

	// calculate boundary weight
	cv::Vec3b color1 = image.at<cv::Vec3b>(cv::Point(xx1, yy1));
	cv::Vec3b color2 = image.at<cv::Vec3b>(cv::Point(xx2, yy2));
	float diff = (abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2])) / 3.f;
	float weight = exp(-bPixelWeight_ * diff);
	boundaries[boundaryIndex].appendWeight(weight);	
	//std::cout << weight << " " << diff << " /// ";
	// save the boundary info & the neighboring segment to each segment
	segments[segment_idx1].updateNeighborInfo(segment_idx2, boundaryIndex);
	segments[segment_idx2].updateNeighborInfo(segment_idx1, boundaryIndex);
}

int Superpixels::getBoundaryIndex(int segment1, int segment2, std::vector<Boundary>& boundaries){

	// check whether the boundary exists

	bool existingBoundary = false;
	int boundaryIndex = -1;

	std::pair<int, int> currSegmentPair;

	if (segment1 < segment2){
		currSegmentPair.first = segment1;
		currSegmentPair.second = segment2;
	}
	else{
		currSegmentPair.first = segment2;
		currSegmentPair.second = segment1;
	}

	for (int ii = 0; ii < boundaries.size(); ++ii ){
		
		if (currSegmentPair == boundaries[ii].getSegmentIndices()){
			existingBoundary = true;
			boundaryIndex = ii;
			break;
		}
	}

	// make a new boundary if it doesn't exist
	if (existingBoundary == false){
		Boundary newBoundary(segment1, segment2);
		boundaryIndex = (int)boundaries.size();
		boundaries.push_back(newBoundary);
	}
	
	return boundaryIndex;
}

static cv::Vec3b randomColor()
{
	return cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
}

void Superpixels::drawBoundaryAndSegementResult(cv::Mat& image, std::vector<Segment>& segments, std::vector<Boundary>& boundaries){

	// draw segment
	for (int idx = 0; idx < segments.size(); idx++){
		cv::Vec3b segColor = randomColor();
		for (int nn = 0; nn < segments[idx].getNumberOfPixels(); ++nn){
			int xx = (int)segments[idx].getPixel(nn).x;
			int yy = (int)segments[idx].getPixel(nn).y;
			image.at<cv::Vec3b>(cv::Point(xx, yy)) = segColor;
		}
		//cv::Vec3b segColor2 = randomColor();
		//for (int nn = 0; nn < segments_[idx].getNumberBoundary(); ++nn){
		//	int bIndex = segments_[idx].getBoundary(nn);
		//	for (int nn = 0; nn < boundaries_[bIndex].getNumberPts(); ++nn){
		//		float xx = boundaries_[bIndex].getPts(nn).x;
		//		float yy = boundaries_[bIndex].getPts(nn).y;

		//		if (xx - (int)xx != 0){
		//			image.at<cv::Vec3b>(cv::Point((int)(xx + 1), (int)yy)) = segColor2;
		//			image.at<cv::Vec3b>(cv::Point((int)(xx), (int)yy)) = segColor2;
		//		}
		//		if (yy - (int)yy != 0){
		//			image.at<cv::Vec3b>(cv::Point((int)xx, (int)(yy + 1))) = segColor2;
		//			image.at<cv::Vec3b>(cv::Point((int)xx, (int)yy)) = segColor2;
		//		}
		//	}
		//	cv::imshow("Superpixel Result", image);
		//	cv::waitKey(0);
		//}
		//std::cout << idx << " ";
		//cv::imshow("Superpixel Result", image);
		//cv::waitKey(0);
	}

	// draw contour
	for (int idx = 0; idx < boundaries.size(); idx++){
		cv::Vec3b segColor = randomColor();
		for (int nn = 0; nn < boundaries[idx].getNumberPts(); ++nn){
			float xx = boundaries[idx].getPts(nn).x;
			float yy = boundaries[idx].getPts(nn).y;
				
			if (xx - (int)xx != 0){
				image.at<cv::Vec3b>(cv::Point((int)(xx + 1), (int)yy)) = segColor;
				image.at<cv::Vec3b>(cv::Point((int)(xx), (int)yy)) = segColor;
			}
			if (yy - (int)yy != 0){
				image.at<cv::Vec3b>(cv::Point((int)xx, (int)(yy+1))) = segColor;
				image.at<cv::Vec3b>(cv::Point((int)xx, (int)yy)) = segColor;
			}
		}
	}
}

bool Superpixels::processTPSeg(const std::string& nameImage, std::vector<Segment>& segments, std::vector<Boundary>& boundaries, cv::Mat& superpixelLabelMap, const std::string& paramDir, bool verbose){

	// param file
	SPSegmentationParameters params = ReadParameters(((std::string)paramDir).append("/").append(fileNameParamsTPSeg_));
	
	cv::Mat image = imread(nameImage, CV_LOAD_IMAGE_COLOR);

	if (image.empty()) {
		std::cout << " ERROR superpiels.cpp : Cannot find the image = " << nameImage << std::endl;
		return false;
	}

	// superpixel
	width_ = image.cols;
	height_ = image.rows;
	cv::Mat resImage, resInfo;
		
	int maxIter = 15;
	while(maxIter > 0){
		
		SPSegmentationEngine engine(params, image);
		engine.ProcessImage();
		engine.PrintPerformanceInfo();

		resImage = engine.GetSegmentedImage();
		resInfo = engine.GetSegmentation();

		// checkSuperpixelValidity
		bool bValid = checkSuperpixelValidity(resInfo);

		if (bValid){
			// save label map
			superpixelLabelMap = resInfo.clone();
			break;
		}
		
		params.superpixelNum = params.superpixelNum - 20;		
		maxIter--;
	}
	//std::cout << "ITER = " << 15 - maxIter << std::endl;
	if (maxIter == 0){
		std::cout << "Serious Fatal Error in Superpixelization" << std::endl;
	}

	// save segment and boundary information
	saveSegmentAndBoundaryFromTPSeg(image, segments, boundaries, resInfo);

	// save the structure of graph: saving indices of n-hop-neighboring superpixels
	saveNNeighborIndex(segments);

	// calculate boundary weight
	calculateBoundaryWeights(image, segments, boundaries);

	if (verbose){
		ShowResizeImage("Segmentation", resImage, 50, 50);
		cv::Mat mColorCopy = image.clone();
		drawBoundaryAndSegementResult(mColorCopy, segments, boundaries);
		for (int ss = 0; ss < segments.size(); ++ss){
			std::string text = std::to_string(ss);
			int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
			double fontScale = 0.3;
			int thickness = 1.5;
			cv::Point2f textOrg(segments[ss].getCenterPoint());
			cv::putText(mColorCopy, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 2);
		}
		cv::imshow("Superpixel Result", mColorCopy);		
		cv::waitKey(0);
	}

	return true;

}

void Superpixels::ShowResizeImage(const std::string& windowName, const cv::Mat& srcImage, int xx, int yy) {

	int widthLimit = 880;
	cv::Mat imageResized = srcImage.clone();

	cv::namedWindow(windowName);
	cv::moveWindow(windowName, xx, yy);

	if (imageResized.cols > widthLimit){

		int heightConv = (int)((float)widthLimit * (float)imageResized.rows / (float)imageResized.cols);
		cv::resize(imageResized, imageResized, cv::Size(widthLimit, heightConv));
		cv::imshow(windowName, imageResized);
	}
	else{
		cv::imshow(windowName, srcImage);
	}
}

void Superpixels::calculateBoundaryWeights(const cv::Mat& image, std::vector<Segment>& segments, std::vector<Boundary>& boundaries) {

	// calculate average point	
	for (int ss = 0; ss < segments.size(); ++ss) {

		std::vector<cv::Point2f> pixels = segments[ss].getPixelVector();
		float sumR = 0;	float sumG = 0;	float sumB = 0;

		for (int pp = 0; pp < pixels.size(); ++pp) {
			cv::Vec3b ppRGB = image.at<cv::Vec3b>((int)pixels[pp].y, (int)pixels[pp].x);
			sumR += (float)ppRGB[2];
			sumG += (float)ppRGB[1];
			sumB += (float)ppRGB[0];
		}

		float pixelSize = (float)pixels.size();
		segments[ss].setAvgColor(cv::Vec3f(sumB / pixelSize, sumG / pixelSize, sumR / pixelSize));
	}

	// boundary weight
	for (int bb = 0; bb < boundaries.size(); ++bb) {

		std::pair<int, int> segIndex = boundaries[bb].getSegmentIndices();
		cv::Vec3f avgColor1 = segments[segIndex.first].getAvgColor();
		cv::Vec3f avgColor2 = segments[segIndex.second].getAvgColor();
	
		float sad = (fabs(avgColor1[0] - avgColor2[0]) + fabs(avgColor1[1] - avgColor2[1]) + fabs(avgColor1[2] - avgColor2[2])) / 3.f;

		float weight = exp(-superpixelWeight_ * sad);
		boundaries[bb].setSpxWeight(weight);
		//std::cout << weight << " " << sad << " // ";
	}
}