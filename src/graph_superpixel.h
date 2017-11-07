#pragma once
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>

class Segment {

public:
	Segment() {
		pixels_.resize(0);
		neighborSegmentIndices_.resize(0);
		boundaryIndices_.resize(0);
		state_ = cv::Mat(3, 3, CV_32FC1);
		homography_ = cv::Mat(3, 3, CV_32FC1);
		neighborSegmentIndexNhops_.resize(5);
		centerPoint_ = cv::Point2f(0, 0);
		avgColor_ = cv::Vec3f(0, 0, 0);
	}

	// Superpixel Graph Construction -------------------------------

	void appendPixel(const int xx, const int yy) {
		cv::Point2f pt((float)xx, (float)yy);
		pixels_.push_back(pt);
	}
	//void updateNeighborInfo(int neighborSegment) {
	//	if (std::find(neighborSegmentIndices_.begin(), neighborSegmentIndices_.end(), neighborSegment) == neighborSegmentIndices_.end()) {
	//		// if doesn't exist in the neighbor list
	//		neighborSegmentIndices_.push_back(neighborSegment);
	//	}
	//}
	void updateNeighborInfo(int neighborSegment, int boundaryIndex) {
		if (std::find(neighborSegmentIndices_.begin(), neighborSegmentIndices_.end(), neighborSegment) == neighborSegmentIndices_.end()) {
			// if doesn't exist in the neighbor list
			neighborSegmentIndices_.push_back(neighborSegment);
			boundaryIndices_.push_back(boundaryIndex);
		}
	}
	int getNumberOfPixels() { return (int)pixels_.size(); }
	cv::Point2f getPixel(int index) { return pixels_[index]; }
	std::vector<cv::Point2f> getPixelVector() { return pixels_; }
	int getNumberOfNeighbor() { return (int)neighborSegmentIndices_.size(); }
	int getNeighbor(int index) { return neighborSegmentIndices_[index]; }
	int getNumberOfBoundary() { return (int)boundaryIndices_.size(); }
	int getBoundary(int index) { return boundaryIndices_[index]; }
	int getLocalIndexOfNeighborIndex(int index) {
		int localIndex = -1;
		for (int ii = 0; ii < neighborSegmentIndices_.size(); ++ii) {
			if (neighborSegmentIndices_[ii] == index) {
				localIndex = ii;
				return localIndex;
			}
		}
		return localIndex;
	}

	// N-hop neighbor configuration
	void setNeighborSegmentIndexNhops(std::set<int> neighborIndex, int n) {
		if (n >= neighborSegmentIndexNhops_.size()) {
			std::cout << " Fatal Error - n is larger than the vector size" << std::endl;
		}
		neighborSegmentIndexNhops_[n] = neighborIndex;
	}
	std::set<int> getNeighborSegmentIndexNhops(int n) {
		return neighborSegmentIndexNhops_[n];
	}

	// Internal property configuration
	void setAvgColor(cv::Vec3f colorIn) {
		avgColor_ = colorIn;
	}
	cv::Vec3f getAvgColor() {
		return avgColor_;
	}

	void setCenterPoint(cv::Point2f point) { centerPoint_ = point; }
	cv::Point2f getCenterPoint() { return centerPoint_; }
	
	//// Seed points and Warped points
	//void setSeedPoints(){
	//	// set seedPoints
	//	seedPositions.resize(4);
	//	seedPositions[0] = cv::Point2f((float)topleftPoint.x, (float)topleftPoint.y);
	//	seedPositions[1] = cv::Point2f((float)bottomrightPoint.x, (float)topleftPoint.y);
	//	seedPositions[2] = cv::Point2f((float)topleftPoint.x, (float)bottomrightPoint.y);
	//	seedPositions[3] = cv::Point2f((float)bottomrightPoint.x, (float)bottomrightPoint.y);
	//}
	//std::vector<cv::Point2f> getSeedPoints(){ return seedPositions; }
	//void setWarpedPoints(int idx, std::vector<cv::Point2f> pts_in){
	//	warpedPositions[idx] = pts_in;
	//}
	//std::vector<cv::Point2f> getWarpedPoints(int idx){
	//	return warpedPositions[idx];
	//}	

	// census patch
	void setCensusPatchTL(cv::Point2i point){ censusPatchTL = point; }
	cv::Point2i getCensusPatchTL(){ return censusPatchTL; }
	void setCensusPatchBR(cv::Point2i point){ censusPatchBR = point; }
	cv::Point2i getCensusPatchBR(){ return censusPatchBR; }

	// vector corner points
	void setVecCornerPts(std::vector<cv::Point2f> vecCornerPts){ vecCornerPts_ = vecCornerPts; }
	std::vector<cv::Point2f> getVecCornerPts(){ return vecCornerPts_; }
	
	// bounding box for LK warping
	void setBboxTLpoint(cv::Point2i point){ bBoxTopLeftPt_ = point; }
	cv::Point2i getBboxTLpoint(){ return bBoxTopLeftPt_; }
	void setBboxBRpoint(cv::Point2i point){ bBoxBottomRightPt_ = point; }
	cv::Point2i getBboxBRpoint() { return bBoxBottomRightPt_; }
	void setBboxSize(cv::Size bsize){ bboxSize_ = bsize; }
	cv::Size getBboxSize(){ return bboxSize_; }
	void setFeatureToTrack(std::vector<cv::Point2f> featureIn){ featuresToTrack_ = featureIn; }
	std::vector<cv::Point2f> getFeatureToTrack(){ return featuresToTrack_; }
	void setImagePatch(cv::Mat patchIn){ imagePatch_ = patchIn; }
	cv::Mat getImagePatch(){ return imagePatch_; }

	//// PMBP ------------------------------------------------------------------------------------------------------------------------//
	//void resizeWarpedPositionSize(int val){
	//	warpedPositions.resize(val);
	//	for (int ii = 0; ii < val; ++ii){
	//		warpedPositions[ii].resize(4);
	//	}
	//}

	// State -----------------------------------------------------------------------------------------------------------------------//
	cv::Mat getState() { return state_; }
	void setState(const cv::Mat& matIn) { state_ = matIn; }
	cv::Mat getHomography() { return homography_; }
	void setHomograph(const cv::Mat& matIn) { homography_ = matIn; }

	cv::Mat getGTstate() { return gt_state_; }
	void setGTstate(const cv::Mat& matIn) { gt_state_ = matIn; }

private:

	// Superpixel Graph Construction -----------------------------------------------------------------------------------------------//
	std::vector<cv::Point2f> pixels_;
	std::vector<int> neighborSegmentIndices_;
	std::vector<int> boundaryIndices_;
	std::vector<std::set<int> > neighborSegmentIndexNhops_;
	cv::Vec3f avgColor_;

	cv::Point2f centerPoint_;

	// LK Warping
	cv::Point2i bBoxTopLeftPt_;
	cv::Point2i bBoxBottomRightPt_;
	cv::Size bboxSize_;
	std::vector<cv::Point2f> featuresToTrack_;
	cv::Mat imagePatch_;

	// census patch
	cv::Point2i censusPatchTL;
	cv::Point2i censusPatchBR;

	// corner points
	std::vector<cv::Point2f> vecCornerPts_;

	//// PMBP ------------------------------------------------------------------------------------------------------------------------//
	//std::vector<cv::Point2f> seedPositions;   // positions of 4 pts from superpixels
	//std::vector<std::vector<cv::Point2f>> warpedPositions;   // positions of 4 pts when warped by the state of the particle

	// State -----------------------------------------------------------------------------------------------------------------------//
	cv::Mat state_;
	cv::Mat homography_;	// not for optimization -- for the final results. during optimization, state are in particles

	// for debugging 
	cv::Mat gt_state_;

};


class Boundary {

public:
	Boundary() {
		ptsPair_.resize(0);
		pts_.resize(0);
	};
	Boundary(int segment1, int segment2) {
		if (segment1 < segment2) {
			segmentIndices_.first = segment1;
			segmentIndices_.second = segment2;
		}
		else {
			segmentIndices_.first = segment2;
			segmentIndices_.second = segment1;
		}
	}

	std::pair<int, int> getSegmentIndices() { return segmentIndices_; }
	void appendPoint(float xx1, float yy1, float xx2, float yy2) {
		cv::Point2f pt1(xx1, yy1);
		cv::Point2f pt2(xx2, yy2);
		cv::Point2f pt_m((xx1 + xx2) / 2.f, (yy1 + yy2) / 2.f);
		std::pair<cv::Point2f, cv::Point2f> currPair = std::pair<cv::Point2f, cv::Point2f>(pt1, pt2);
		ptsPair_.push_back(currPair);
		pts_.push_back(pt_m);
	}
	void appendWeight(float weight){
		ptsWeight_.push_back(weight);
	}
	int getNumberPts() { return (int)pts_.size(); }
	cv::Point2f getPts(int index) { return pts_[index]; }
	std::vector<cv::Point2f> getPixelVector() { return pts_; }
	std::vector<float> getPixelWeight() { return ptsWeight_; }
	void setSpxWeight(float spxWeight) { spxWeight_ = spxWeight; }
	float getSpxWeight() { return spxWeight_; }
	
private:
	std::pair<int, int> segmentIndices_;
	std::vector<std::pair<cv::Point2f, cv::Point2f> > ptsPair_;
	std::vector<cv::Point2f> pts_;
	std::vector<float> ptsWeight_;
	float spxWeight_;

};