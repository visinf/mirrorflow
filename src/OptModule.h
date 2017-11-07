#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>
#include <random>
#include "graph_superpixel.h"
#include "opt_util.h"
#include <stdint.h>			// writeFlowField
#include <png++/png.hpp>	// writeFlowField

class OptModule {

public:

	// general
	OptModule(std::string nameOptParamFile){
		fileNameParams_ = nameOptParamFile;
		identityMat_ = (cv::Mat_<float>(3, 3) << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f);
	}

	// -------------------------------------------------------------------

	bool initializeVariables(
		const std::string& nameCurrFrame, const std::string& nameNextFrame,
		const cv::Mat& initflowX_F, const cv::Mat& initflowX_B,
		const cv::Mat& initflowY_F, const cv::Mat& initflowY_B,
		std::vector<Segment>& segments_F, std::vector<Segment>& segments_B,
		std::vector<Boundary>& boundaries_F, std::vector<Boundary>& boundaries_B,
		const cv::Mat& superpixelLabelMap_F, const cv::Mat& superpixelLabelMap_B,
		const std::string& dirRes, const std::string& prefix);	

	bool solve_QPBO_LAE();

	cv::Mat visualizeFlowMap(OptView view);
	void WriteFlowField(const std::string& filename_f, const std::string& filename_b, const std::string& filename_occ_f, const std::string& filename_occ_b);
	void ReadFlowField(const std::string& file_name, cv::Mat& flowX, cv::Mat& flowY);
	void FreeMemory();

private:

	void readParams(const std::string& fileName);
	void PrintParameter();	
	void defineRegionGroups_square(OptView view, int n_superpixels);
	std::set<int> getDifferenceBetweenSets(const std::set<int>& set, const std::set<int>& subset);
	void visualizeRegionGroups(OptView view);
	void initializeState_lae(OptView view);
	cv::Mat GetHomographyStateFromFlowMap(OptView view, int index); 
	
	void optimize_O_lae(OptView view, int oo_iter = 0);	
	
	void optimize_H_qpbo_lae(const OptView view);
	void propagation_qpbo_lae(const OptView view, const std::set<int>& setCenGrpSpx, const std::set<int>& setBndGrpSpx, const std::set<int>& setExpGrpSpx, const std::set<int>& setBnds, const int n_samples);
	void randomize_qpbo_lae(const OptView view, const std::set<int>& setCenGrpSpx, const std::set<int>& setBndGrpSpx, const std::set<int>& setExpGrpSpx, const std::set<int>& setBnds, const int n_samples, bool verbose = false);
	void runQPBO_lae(const OptView view,
		const std::set<int>& setCenGrpSpx,
		const std::set<int>& setBndGrpSpx,
		const std::set<int>& setBnds,
		const std::vector<cv::Mat>& setStates);

	std::vector<cv::Mat> collectRandomNstates(OptView view, const std::set<int>& set_in, int n, bool refine = false, bool add_noise = false, float noise_scale = 0, float noise_trans = 0);
	std::vector<cv::Mat> collectRandomNstatesFromOpposite(OptView view, const std::set<int>& set_in, int n, bool refine = false);
	std::vector<cv::Mat> collectRandomNstatesFusingStateAround(OptView view, const std::set<int>& set_in, int n, float ratio);
	std::vector<cv::Mat> collectRandomNstatesUsingConsistency(OptView view, const std::set<int>& set_in, int n_samples);
	cv::Mat getRandomHomographyFromAround(OptView view, int index, float ratio);

	void updateFlowFromOptRes(const OptView view, const int idx_ss);
	cv::Mat computeHomography_stable(const std::vector<cv::Point2f>& srcP, const std::vector<cv::Point2f>& dstP, bool bRansac = false, float thres = 2.0f);

	cv::Mat getPerturbedState(OptView view, int idx_ss, const cv::Mat& state_in, float noise_scale, float noise_trans);
	cv::Mat getRefinedState(OptView view, const int idx_ss, const cv::Mat& state_in);
	bool checkSimilarState(const std::vector<cv::Mat>& setState, const cv::Mat& seedState);
	bool checkSimilarState_superpixelwise(const OptView view, const std::vector<cv::Mat>& setState, const cv::Mat& seedState, const int idx_ss);

	// local region groups
	std::vector<std::vector<std::set<int> > > centerRegionGroup1hop_[2];
	std::vector<std::vector<std::set<int> > > centerRegionGroup2hop_[2];
	std::vector<std::vector<std::set<int> > > boundaryRegionGroup1hop_[2];
	std::vector<std::vector<std::set<int> > > boundaryRegionGroup2hop_[2];
	std::vector<std::vector<std::set<int> > > expansionRegionGroup1hop_[2];
	std::vector<std::vector<std::set<int> > > expansionRegionGroup2hop_[2];
	std::vector<std::vector<std::set<int> > > boundaryGroup1hop_[2];
	std::vector<std::vector<std::set<int> > > boundaryGroup2hop_[2];

	std::set<int> allSpxGroup_[2];
	std::set<int> boundaryGroup_[2];

	// cost and terms
	float dataCost(OptView view, const int idx_spx, const cv::Mat& state, bool verbose = false);
	float dataTerm(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state, const cv::Point2i& tlPoint, const cv::Point2i& brPoint);
	float dataTerm_cont(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state, const cv::Point2i& tlPoint, const cv::Point2i& brPoint);
	float consistencyTerm(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state);
	float symmetryTerm(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state);
	float pairwiseCost(OptView view, const int idx_spx1, const cv::Mat& state1, const int idx_spx2, const cv::Mat& state2, const int boundaryIndex);
	float pairwiseCost_co_hi(OptView view, const int idx_spx1, const cv::Mat& state1, const int idx_spx2, const cv::Mat& state2, const int boundaryIndex);

	float data_ternary_grad(OptView source, OptView target, const cv::Point2f& pixel, cv::Point2f& tformPixel, const cv::Point2i& tlPoint, int wPatch, std::vector<float>& dstPatchVal, std::vector<cv::Point2f>& tformDstPatch);
	float data_ternary_grad_rev1(OptView source, OptView target, std::vector<bool> oobMask, const cv::Point2f& pixel, const cv::Point2f& tformPixel, const cv::Point2i& tlPoint, int wPatch, std::vector<float>& dstPatchVal, std::vector<cv::Point2f>& tformDstPatch);
	float data_intensity_grad(OptView source, OptView target, const cv::Point2f& pixel, const cv::Point2f& tformPixel);
	
	std::vector<float> dataCost_occ(OptView view);
	std::vector<float> dataTerm_occ_cont(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state, const cv::Point2i& tlPoint, const cv::Point2i& brPoint);
	std::vector<float> consistencyTerm_occ(OptView view);
	std::vector<float> symmetryTerm_occ(OptView view);
	void updateConsistencyEnergyMap(OptView view, const std::set<int>& set_in);

	// image processing
	void getSignedGradientImage(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY);
	cv::Mat getCensusImage(const cv::Mat& image);
	cv::Mat getTernaryImage(const cv::Mat& image);
	void getTernaryImage_continuous(const cv::Mat& image, std::vector<std::vector<float> >& ternary_mat_cont);
	void getTransformedPoints(const std::vector<cv::Point2f>& srcPoints, const cv::Mat& state, std::vector<cv::Point2f>& dstPoints);
	void getTransformedPoint(const cv::Point2f& srcPoints, const cv::Mat& state, cv::Point2f& dstPoints);
	bool isInside(OptView view, float x, float y) const;
	bool isStateValid(OptView view, int index, const cv::Mat& state);

	inline float ternary_sigmoid(float val) const;
	inline int hammingDistanceTernary(double in1, double in2) const;
	inline void hammingDistanceTernary_cont(std::vector<float> ter1, std::vector<float> ter2, float& dist, int& n_oob);
	double getTernaryValfromPatch(OptView view, const cv::Point2f& srcP, const cv::Point2i& tlPoint, int wPatch, const std::vector<float>& dstPatchVal, const std::vector<cv::Point2f>& tformDstPatch, int ternarySize) const;
	std::vector<float> getTernaryValfromPatch_cont(OptView view, const cv::Point2f& srcP, const cv::Point2i& tlPoint, int wPatch, const std::vector<float>& dstPatchVal, const std::vector<cv::Point2f>& tformDstPatch) const;
	void getInterpolatedPixel_single(const cv::Mat& image, float x, float y, float& val) const;
	void getInterpolatedPixel_single_float(const cv::Mat& image, float x, float y, float& val) const;
	void projectionIntoTheOtherView(OptView view);
	bool findCorrespondencebyWarping(OptView view, int index, std::vector<cv::Point2f>& srcP, std::vector<cv::Point2f>& dstP);

	// output
	void printOverallEnergy(OptView view);
	float getEnergyDataTerm(OptView view);
	float getEnergyPairwiseTerm(OptView view);
	float getEnergyConsistencyTerm(OptView view);
	float getEnergySymmetryTerm(OptView view);
	float getEnergyOccPairwiseTerm(OptView view);
	void visualizeEnergy(OptView view, bool init = false);

	// param
	std::string fileNameParams_ = "";
	OptParameters params_;

	// state
	std::vector<cv::Point2f> seedPts_;
	std::vector<Segment>* segments_[2]; 
	std::vector<Boundary>* boundaries_[2];
	cv::Mat superpixelLabelMap_[2];
	cv::Mat identityMat_;

	// image processing
	cv::Mat images_mat_[2];
	cv::Mat gray_mat_[2];
	cv::Mat gradX_mat_[2];
	cv::Mat gradY_mat_[2];
	cv::Mat census_mat_[2];
	cv::Mat ternary_mat_[2];
	cv::Mat median_mat_[2];
	std::vector<std::vector<float> > ternary_mat_cont[2];
	
	cv::Mat flowX_[2];
	cv::Mat flowY_[2];
	cv::Mat occMask_[2];
	cv::Mat projectionMask_[2];
	cv::Mat consistencyEnergy_[2];

	int width_;
	int height_;
	int featureChannel_;
	int ternaryHalfWidth_;
	int ternaryPatchSize_;
	int ternaryOobCode_;	// out of bound

	// logfile
	std::ofstream log_file_;

	// temp
	float cnnUnaryWeight = 60.f;
};