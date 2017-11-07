#include "OptModule.h"
#include "colorcode.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <map>
#include "gco/GCoptimization.h"
#include "qpbo/QPBO.h"
#include <omp.h>

float LC[10][5] =
{ { 0, 2, 49, 54, 149 },
{ 2, 4, 69, 117, 180 },
{ 4, 6, 116, 173, 209 },
{ 6, 8, 171, 217, 233 },
{ 8, 10, 224, 243, 248 },
{ 10, 12, 254, 224, 144 },
{ 12, 14, 253, 174, 97 },
{ 14, 17, 244, 109, 67 },
{ 17, 20, 215, 48, 39 },
{ 20, 1000000000.0, 165, 0, 38 } };

float LC_log[10][5] =
{ { 0, 0.0625, 49, 54, 149 },
{ 0.0625, 0.125, 69, 117, 180 },
{ 0.125, 0.25, 116, 173, 209 },
{ 0.25, 0.5, 171, 217, 233 },
{ 0.5, 1, 224, 243, 248 },
{ 1, 2, 254, 224, 144 },
{ 2, 4, 253, 174, 97 },
{ 4, 8, 244, 109, 67 },
{ 8, 16, 215, 48, 39 },
{ 16, 1000000000.0, 165, 0, 38 } };

struct ForDataFn {
	int numLab;
	float *data;
};

struct ForSmoothFn {
	int numP;
	int numLab;
	float *data;
};

float smoothFn(int p1, int p2, int s1, int s2, void *data)
{
	ForSmoothFn *myData = (ForSmoothFn *)data;
	int numLab = myData->numLab;
	int numP = myData->numP;

	int idx = (p1 + numP * p2) * numLab * numLab + s1 * numLab + s2;

	return(myData->data[idx]);
}

float dataFn(int p, int s, void *data)
{
	ForDataFn *myData = (ForDataFn *)data;
	int numLab = myData->numLab;

	return(myData->data[p*numLab + s]);
}

bool OptModule::initializeVariables(const std::string& nameCurrFrame, const std::string& nameNextFrame,
	const cv::Mat& initflowX_F, const cv::Mat& initflowX_B,
	const cv::Mat& initflowY_F, const cv::Mat& initflowY_B,
	std::vector<Segment>& segments_F, std::vector<Segment>& segments_B,
	std::vector<Boundary>& boundaries_F, std::vector<Boundary>& boundaries_B,
	const cv::Mat& superpixelLabelMap_F, const cv::Mat& superpixelLabelMap_B,
	const std::string& dirRes, const std::string& prefix) {

	// read params
	readParams(fileNameParams_);
	params_.prefix = prefix;

	// logging
	if (params_.verbose) {
		std::stringstream ss_dir;
		ss_dir << (std::string(dirRes)) << "log_file_" << prefix << ".txt";
		log_file_.open(ss_dir.str());
	}
	params_.dirRes = dirRes;

	// initialize
	images_mat_[currFrame] = cv::imread(nameCurrFrame);
	images_mat_[nextFrame] = cv::imread(nameNextFrame);

	// gray
	cv::cvtColor(images_mat_[currFrame], gray_mat_[currFrame], CV_BGR2GRAY);
	cv::cvtColor(images_mat_[nextFrame], gray_mat_[nextFrame], CV_BGR2GRAY);

	// resolution
	width_ = images_mat_[currFrame].cols;
	height_ = images_mat_[currFrame].rows;

	// census
	ternaryHalfWidth_ = 3;
	ternaryPatchSize_ = (2 * ternaryHalfWidth_ + 1)*(2 * ternaryHalfWidth_ + 1);
	ternaryOobCode_ = -100;

	// grad
	getSignedGradientImage(images_mat_[currFrame], gradX_mat_[currFrame], gradY_mat_[currFrame]);
	getSignedGradientImage(images_mat_[nextFrame], gradX_mat_[nextFrame], gradY_mat_[nextFrame]);

	// census
	census_mat_[currFrame] = getCensusImage(images_mat_[currFrame]);
	census_mat_[nextFrame] = getCensusImage(images_mat_[nextFrame]);

	// ternary
	ternary_mat_[currFrame] = getTernaryImage(images_mat_[currFrame]);
	ternary_mat_[nextFrame] = getTernaryImage(images_mat_[nextFrame]);

	// median
	cv::medianBlur(images_mat_[currFrame], median_mat_[currFrame], 3);
	cv::medianBlur(images_mat_[nextFrame], median_mat_[nextFrame], 3);

	// ternary - continuous
	getTernaryImage_continuous(images_mat_[currFrame], ternary_mat_cont[currFrame]);
	getTernaryImage_continuous(images_mat_[nextFrame], ternary_mat_cont[nextFrame]);

	// flow
	flowX_[currFrame] = initflowX_F.clone();
	flowY_[currFrame] = initflowY_F.clone();
	flowX_[nextFrame] = initflowX_B.clone();
	flowY_[nextFrame] = initflowY_B.clone();

	// occlusion mask
	occMask_[currFrame] = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(255));
	occMask_[nextFrame] = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(255));

	// projection mask
	projectionMask_[currFrame] = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(0));
	projectionMask_[nextFrame] = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(0));

	// graph-superpixel
	segments_[currFrame] = &segments_F;
	segments_[nextFrame] = &segments_B;
	boundaries_[currFrame] = &boundaries_F;
	boundaries_[nextFrame] = &boundaries_B;

	// superpixel label map
	superpixelLabelMap_[currFrame] = superpixelLabelMap_F.clone();
	superpixelLabelMap_[nextFrame] = superpixelLabelMap_B.clone();

	// consistency Energy map	
	consistencyEnergy_[currFrame] = cv::Mat(height_, width_, CV_64FC1, 0.f);
	consistencyEnergy_[nextFrame] = cv::Mat(height_, width_, CV_64FC1, 0.f);

	if (params_.bidirectional == false){
		params_.lambda_consistency = 0.f;
		params_.lambda_symmetry = 0.f;
		params_.lambda_occPott = 0.f;
	}

	// set seed points for checking similar state
	{
		float x_center = width_ / 2.f;
		float y_center = height_ / 2.f;
		float patch_size_half = 15.f;
		seedPts_.push_back(cv::Point2f(x_center - patch_size_half, y_center - patch_size_half));
		seedPts_.push_back(cv::Point2f(x_center + patch_size_half, y_center - patch_size_half));
		seedPts_.push_back(cv::Point2f(x_center - patch_size_half, y_center + patch_size_half));
		seedPts_.push_back(cv::Point2f(x_center + patch_size_half, y_center + patch_size_half));
	}

	//PrintParameter_lae();

	return true;
}

bool OptModule::solve_QPBO_LAE() {

#ifdef _OPENMP
	double begin_op_t = omp_get_wtime();
#else
	clock_t begin_t, poll_t;
	begin_t = clock();
#endif
	if (params_.verbose){
		log_file_ << std::endl << "Initialization  ----------------------" << std::endl;
	}

	// define Region Groups 
	if (params_.verbose){
		log_file_ << " - defineRegionGroups" << std::endl;
	}
	defineRegionGroups_square(currFrame, (int)(segments_[currFrame]->size()));
	if (params_.bidirectional == true){
		defineRegionGroups_square(nextFrame, (int)(segments_[nextFrame]->size()));
	}

	// initialize state
	if (params_.verbose){
		log_file_ << " - initializeState" << std::endl;
	}
	initializeState_lae(currFrame);
	initializeState_lae(nextFrame);

	// calculate projection mat
	if (params_.verbose){
		log_file_ << " - projectionIntoTheOtherView" << std::endl;
	}
	projectionIntoTheOtherView(currFrame);
	projectionIntoTheOtherView(nextFrame);
	optimize_O_lae(currFrame);
	optimize_O_lae(nextFrame);

	if (params_.verbose){
		log_file_ << " - printOverallEnergy" << std::endl;
	}
	printOverallEnergy(currFrame);
	if (params_.bidirectional == true){
		printOverallEnergy(nextFrame);
		log_file_ << std::endl;
	}
	
	// outer iteration
	for (int oo = 0; oo < params_.n_outer_iterations; ++oo) {

		// Optimize motion
		if (params_.verbose){
			std::cout << std::endl << "Outer Iteration " << oo + 1 << "/" << params_.n_outer_iterations << "----------------------" << std::endl;
			log_file_ << std::endl << "Outer Iteration " << oo + 1 << "/" << params_.n_outer_iterations << "----------------------" << std::endl;
			log_file_ << "  Opt H_forward" << std::endl;
		}
		optimize_H_qpbo_lae(currFrame);

		if (params_.bidirectional == true){

			// projection to the other view
			projectionIntoTheOtherView(currFrame);

			// Optimize occlusion
			if (params_.verbose){
				log_file_ << "  Opt O_forward" << std::endl;
			}
			optimize_O_lae(nextFrame, oo);

			// Optimize motion
			if (params_.verbose){
				log_file_ << "  Opt H_backward" << std::endl;
			}
			optimize_H_qpbo_lae(nextFrame);

			// projection to the other view
			projectionIntoTheOtherView(nextFrame);

			// Optimize occlusion
			if (params_.verbose){
				log_file_ << "  Opt O_backward" << std::endl;
			}
			optimize_O_lae(currFrame, oo);
		}

		// when doing the superpixel refinement, it needs to update the superpixelLabelMap_ !!!!		
		printOverallEnergy(currFrame);
		if (params_.bidirectional == true){
			printOverallEnergy(nextFrame);
		}
		log_file_ << std::endl;


#ifdef _OPENMP
		double poll_op_t = omp_get_wtime();
		log_file_ << "  Elapsed Time: " << (poll_op_t - begin_op_t) << " (s)" << std::endl;
#else
		poll_t = clock();
		log_file_ << "  Elapsed Time: " << ((poll_t - begin_t) / CLOCKS_PER_SEC) << " (s)" << std::endl;
#endif




		//{
		//	std::string flowName_f = params_.dirRes + "iter_" + std::to_string(oo) + "_" + params_.prefix + "_10.png";
		//	std::string flowName_b = params_.dirRes + "iter_" + std::to_string(oo) + "_" + params_.prefix + "_10_b.png";
		//	std::string occName_f = params_.dirRes + "occ_iter_" + std::to_string(oo) + "_" + params_.prefix + "_10.png";
		//	std::string occName_b = params_.dirRes + "occ_iter_" + std::to_string(oo) + "_" + params_.prefix + "_10_b.png";
		//	WriteFlowField(flowName_f, flowName_b, occName_f, occName_b);
		//}


		if (params_.verbose && oo == params_.n_outer_iterations - 1){
			//if (params_.verbose){
			//cv::Mat currFlow_f = visualizeFlowMap(currFrame);
			////cv::imshow("flow_forward", currFlow_f);
			//std::stringstream oo_s;
			//oo_s << oo;

			//cv::imwrite(std::string(params_.dirRes + "res_" + params_.prefix + "_flow1" + "_iter_" + oo_s.str() + ".png"), currFlow_f);
			//if (params_.bidirectional == true){
			//	cv::Mat currFlow_b = visualizeFlowMap(nextFrame);
			//	cv::Mat occ_f = occMask_[currFrame];
			//	cv::Mat occ_b = occMask_[nextFrame];
			//	//cv::imshow("flow_backward", currFlow_b);
			//	//cv::imshow("occ_forward", occ_f);
			//	//cv::imshow("occ_backward", occ_b);
			//	cv::imwrite(std::string(params_.dirRes + "res_" + params_.prefix + "_flow2" + "_iter_" + oo_s.str() + ".png"), currFlow_b);
			//	cv::imwrite(std::string(params_.dirRes + "res_" + params_.prefix + "_occ1" + "_iter_" + oo_s.str() + ".png"), occ_f);
			//	cv::imwrite(std::string(params_.dirRes + "res_" + params_.prefix + "_occ2" + "_iter_" + oo_s.str() + ".png"), occ_b);
			//	cv::imwrite(std::string(params_.dirRes + "res_" + params_.prefix + "_warp1" + "_iter_" + oo_s.str() + ".png"), projectionMask_[currFrame]);
			//	cv::imwrite(std::string(params_.dirRes + "res_" + params_.prefix + "_warp2" + "_iter_" + oo_s.str() + ".png"), projectionMask_[nextFrame]);
			//}


#ifdef _OPENMP
			log_file_ << "  Elapsed Time: " << (poll_op_t - begin_op_t) << " (s)" << std::endl;
#else			
			std::cout << "  Elapsed Time: " << ((poll_t - begin_t) / CLOCKS_PER_SEC) << " (s)" << std::endl;
#endif

			//// energy visualization
			//visualizeEnergy(currFrame);
			//visualizeEnergy(nextFrame);

		}
	}

	//visualizeEnergy(currFrame, false);
	//visualizeEnergy(nextFrame, false);

	if (params_.verbose){
		//cv::imwrite(params_.dirRes + "forwardFlow.png", visualizeFlowMap(currFrame));
		//if (params_.bidirectional == true){
		//	cv::imwrite(params_.dirRes + "backwardFlow.png", visualizeFlowMap(nextFrame));
		//	cv::imwrite(params_.dirRes + "forwardOcc.png", occMask_[currFrame]);
		//	cv::imwrite(params_.dirRes + "backwardOcc.png", occMask_[nextFrame]);			
		//}
		log_file_.close();
	}
	return true;
}

void OptModule::defineRegionGroups_square(OptView view, int n_superpixels) {

	int spx_len = (int)(sqrtf((float)width_ * (float)height_ / (float)params_.superpixelNum));
	int region_w_est = (int)(width_ / (float)(params_.region_width));
	int region_h_est = (int)(height_ / (float)(params_.region_height));
	float overlap_ratio = params_.region_overlap_ratio;

	int nCellsHorizontal = (int)(((float)width_ - region_w_est*overlap_ratio) / (region_w_est - region_w_est*overlap_ratio) + 0.5f);
	int nCellsVertical = (int)(((float)height_ - region_h_est*overlap_ratio) / (region_h_est - region_h_est*overlap_ratio) + 0.5f);

	float region_w = (float)width_ / ((float)nCellsHorizontal - overlap_ratio*(nCellsHorizontal - 1.f));
	float region_h = (float)height_ / ((float)nCellsVertical - overlap_ratio*(nCellsVertical - 1.f));

	{
		int nCells_1d = (int)(1.f / (1.f - overlap_ratio) + 0.99f);
		int nGroups = nCells_1d * nCells_1d;

		centerRegionGroup2hop_[view].resize(nGroups);
		boundaryRegionGroup2hop_[view].resize(nGroups);
		expansionRegionGroup2hop_[view].resize(nGroups);
		boundaryGroup2hop_[view].resize(nGroups);

		std::vector<int> w_bins;
		std::vector<int> h_bins;
		w_bins.resize(nCellsHorizontal + 1);
		h_bins.resize(nCellsVertical + 1);
		w_bins[0] = -spx_len;
		h_bins[0] = -spx_len;
		w_bins[nCellsHorizontal] = spx_len;
		h_bins[nCellsVertical] = spx_len;
		for (int n_ww = 1; n_ww < nCellsHorizontal; ++n_ww) {
			w_bins[n_ww] = (int)(n_ww * region_w * (1.f - overlap_ratio) + 0.5f);
		}
		for (int n_hh = 1; n_hh < nCellsVertical; ++n_hh) {
			h_bins[n_hh] = (int)(n_hh * region_h * (1.f - overlap_ratio) + 0.5f);
		}

		//// sanity check
		//{
		//	for (int n_ww = 0; n_ww < nCellsHorizontal + 1; ++n_ww) {
		//		std::cout << w_bins[n_ww] << " ";
		//	}
		//	std::cout << std::endl;
		//	for (int n_hh = 0; n_hh < nCellsVertical + 1; ++n_hh) {
		//		std::cout << h_bins[n_hh] << " ";
		//	}
		//	std::cout << std::endl;
		//}

		for (int n_hh = 0; n_hh < nCellsVertical; ++n_hh) {

			int h_str = int(h_bins[n_hh] + spx_len);
			int h_end = int(h_bins[n_hh] + region_h - spx_len);

			if (n_hh == 0){
				h_str = 0;
			}
			if (n_hh == nCellsVertical - 1){
				h_end = height_;
			}

			for (int n_ww = 0; n_ww < nCellsHorizontal; ++n_ww) {

				int w_str = w_bins[n_ww] + spx_len;
				int w_end = w_bins[n_ww] + region_w - spx_len;

				if (n_ww == 0){
					w_str = 0;
				}
				if (n_ww == nCellsHorizontal - 1){
					w_end = width_;
				}

				// collect the set of superpixel indices
				std::set<int> spx_indices;
				for (int pp_hh = h_str; pp_hh < h_end; ++pp_hh){
					for (int pp_ww = w_str; pp_ww < w_end; ++pp_ww){
						int idxSp = (int)(superpixelLabelMap_[view].at<unsigned short>(pp_hh, pp_ww));
						spx_indices.insert(idxSp);
					}
				}

				// calculate index group
				int idxGroup = (n_hh % nCells_1d) * nCells_1d + n_ww % nCells_1d;

				// save center region
				centerRegionGroup2hop_[view][idxGroup].push_back(spx_indices);

				// find a set of 1-hop neighbor
				std::set<int> spx_1hop_neighbor;
				for (auto iter = spx_indices.begin(); iter != spx_indices.end(); ++iter){
					int seed = *iter;
					int neighborNum = segments_[view]->at(seed).getNumberOfNeighbor();
					for (int nnn = 0; nnn < neighborNum; ++nnn) {
						int idxNeighborSpx = segments_[view]->at(seed).getNeighbor(nnn);
						spx_1hop_neighbor.insert(idxNeighborSpx);
					}
				}

				// find an expansion group
				std::set<int> spx_ExpRegion;
				for (auto iter = spx_indices.begin(); iter != spx_indices.end(); ++iter){
					spx_ExpRegion.insert(*iter);
				}
				for (auto iter = spx_1hop_neighbor.begin(); iter != spx_1hop_neighbor.end(); ++iter){
					spx_ExpRegion.insert(*iter);
				}
				expansionRegionGroup2hop_[view][idxGroup].push_back(spx_ExpRegion);

				// find a boundary group
				std::set<int> spx_BndRegion = getDifferenceBetweenSets(spx_ExpRegion, spx_indices);
				boundaryRegionGroup2hop_[view][idxGroup].push_back(spx_BndRegion);

				// calculate boundary-label-set;
				std::set<int> setBoundaries;
				for (auto iter = spx_ExpRegion.begin(); iter != spx_ExpRegion.end(); ++iter) {
					int n_boundary = segments_[view]->at(*iter).getNumberOfBoundary();
					for (int nn = 0; nn < n_boundary; ++nn) {
						int idx_boundary = segments_[view]->at(*iter).getBoundary(nn);
						int idx_seg1 = boundaries_[view]->at(idx_boundary).getSegmentIndices().first;
						int idx_seg2 = boundaries_[view]->at(idx_boundary).getSegmentIndices().second;

						auto iter_find1 = spx_ExpRegion.find(idx_seg1);
						if (iter_find1 == spx_ExpRegion.end()) {	// when it is not in the set.
							continue;
						}

						auto iter_find2 = spx_ExpRegion.find(idx_seg2);
						if (iter_find2 == spx_ExpRegion.end()) {	// when it is not in the set.
							continue;
						}

						setBoundaries.insert(idx_boundary);
					}
				}
				boundaryGroup2hop_[view][idxGroup].push_back(setBoundaries);

				// sanity check
				for (auto iter_pp = spx_indices.begin(); iter_pp != spx_indices.end(); ++iter_pp) {
					auto iter_find = spx_BndRegion.find(*iter_pp);
					if (iter_find != spx_BndRegion.end()) {	// when it is in the set.
						std::cout << " ERROR OptModlue.cpp : the two set is not mutually exclusive " << std::endl;
					}
				}
			}
		}
	}

	
	//visualizeRegionGroups(view);


	// set of all superpixels
	for (int ii = 0; ii < segments_[view]->size(); ++ii){
		allSpxGroup_[view].insert(ii);
	}
	// set of all superpixels
	for (int ii = 0; ii < boundaries_[view]->size(); ++ii){
		boundaryGroup_[view].insert(ii);
	}

}

std::set<int> OptModule::getDifferenceBetweenSets(const std::set<int>& set, const std::set<int>& subset) {

	std::set<int> resSet;

	for (auto iter = set.begin(); iter != set.end(); ++iter) {

		auto iter_find = subset.find(*iter);
		if (iter_find != subset.end()) {	// when it is in the set.
			continue;
		}

		resSet.insert(*iter);
	}

	return resSet;
}

void OptModule::visualizeRegionGroups(OptView view) {

	cv::Mat imageSpx;
	images_mat_[view].copyTo(imageSpx);

	// draw segment
	for (int idx = 0; idx < segments_[view]->size(); idx++) {
		cv::Vec3b segColor = cv::Vec3b(rand() & 30, rand() & 30, rand() & 30);
		for (int nn = 0; nn < segments_[view]->at(idx).getNumberOfPixels(); ++nn) {
			int xx = (int)segments_[view]->at(idx).getPixel(nn).x;
			int yy = (int)segments_[view]->at(idx).getPixel(nn).y;
			imageSpx.at<cv::Vec3b>(cv::Point(xx, yy)) = segColor;
		}
	}

	// draw contour
	for (int idx = 0; idx < boundaries_[view]->size(); idx++) {
		cv::Vec3b segColor = cv::Vec3b(rand() & 55 + 70, rand() & 55 + 70, rand() & 55 + 70);
		for (int nn = 0; nn < boundaries_[view]->at(idx).getNumberPts(); ++nn) {
			float xx = boundaries_[view]->at(idx).getPts(nn).x;
			float yy = boundaries_[view]->at(idx).getPts(nn).y;

			if (xx - (int)xx != 0) {
				imageSpx.at<cv::Vec3b>(cv::Point((int)(xx + 1), (int)yy)) = segColor;
				imageSpx.at<cv::Vec3b>(cv::Point((int)(xx), (int)yy)) = segColor;
			}
			if (yy - (int)yy != 0) {
				imageSpx.at<cv::Vec3b>(cv::Point((int)xx, (int)(yy + 1))) = segColor;
				imageSpx.at<cv::Vec3b>(cv::Point((int)xx, (int)yy)) = segColor;
			}
		}
	}

	cv::imshow("imgSpx", imageSpx);
	cv::waitKey(0);

	// visualize the superpixels
	cv::Vec3b colorCnt = cv::Vec3b(100, 255, 100);
	cv::Vec3b colorBnd = cv::Vec3b(100, 255, 255);

	cv::Mat dispImg;
	imageSpx.copyTo(dispImg);

	for (int gg = 0; gg < centerRegionGroup2hop_[view].size(); ++gg) {
		for (int ii = 0; ii < centerRegionGroup2hop_[view][gg].size(); ++ii) {

			for (auto iter = centerRegionGroup2hop_[view][gg][ii].begin(); iter != centerRegionGroup2hop_[view][gg][ii].end(); ++iter) {
				int idx = *iter;
				for (int nn = 0; nn < segments_[view]->at(idx).getNumberOfPixels(); ++nn) {
					int xx = (int)segments_[view]->at(idx).getPixel(nn).x;
					int yy = (int)segments_[view]->at(idx).getPixel(nn).y;
					dispImg.at<cv::Vec3b>(cv::Point(xx, yy)) = colorCnt;
				}
			}

			for (auto iter = boundaryRegionGroup2hop_[view][gg][ii].begin(); iter != boundaryRegionGroup2hop_[view][gg][ii].end(); ++iter) {
				int idx = *iter;
				for (int nn = 0; nn < segments_[view]->at(idx).getNumberOfPixels(); ++nn) {
					int xx = (int)segments_[view]->at(idx).getPixel(nn).x;
					int yy = (int)segments_[view]->at(idx).getPixel(nn).y;
					dispImg.at<cv::Vec3b>(cv::Point(xx, yy)) = colorBnd;
				}
			}

			cv::imshow("imgSpx", dispImg);
			cv::waitKey(0);
		}
	}

	cv::imshow("visualizingRegions2", dispImg);
	cv::waitKey(0);

}

void OptModule::visualizeEnergy(OptView view, bool init) {

	int n_segments = (int)(segments_[view]->size());
	int n_boundary = (int)(boundaries_[view]->size());
	OptView source = view;
	OptView target = OtherView(source);

	// Data term
	cv::Mat dataCostMat = cv::Mat(height_, width_, CV_64FC1, 0.f);
	{
		for (int ss = 0; ss < n_segments; ++ss) {
			std::vector<cv::Point2f> pixels = segments_[view]->at(ss).getPixelVector();
			cv::Point2i tlPoint = segments_[view]->at(ss).getCensusPatchTL();
			cv::Point2i brPoint = segments_[view]->at(ss).getCensusPatchBR();
			cv::Mat state = segments_[view]->at(ss).getState();

			std::vector<cv::Point2f> tformedPixels;

			getTransformedPoints(pixels, state, tformedPixels);
			int sizePixels = (int)pixels.size();

			// perspectiveCensus, create patch 
			int wPatch = brPoint.x - tlPoint.x + 2 * ternaryHalfWidth_ + 1;
			int hPatch = brPoint.y - tlPoint.y + 2 * ternaryHalfWidth_ + 1;
			std::vector<cv::Point2f> dstPatch, tformDstPatch;
			dstPatch.resize(wPatch*hPatch);
			tformDstPatch.resize(wPatch*hPatch);
			std::vector<bool> oobMask;
			oobMask.resize(wPatch*hPatch, true);
			int count_pp = 0;
			for (int hh = tlPoint.y - ternaryHalfWidth_; hh <= brPoint.y + ternaryHalfWidth_; ++hh) {
				for (int ww = tlPoint.x - ternaryHalfWidth_; ww <= brPoint.x + ternaryHalfWidth_; ++ww) {
					dstPatch[count_pp] = cv::Point2f((float)ww, (float)hh);
					count_pp++;
				}
			}

			// get transformed pts
			getTransformedPoints(dstPatch, state, tformDstPatch);

			// color interpolation
			std::vector<float> dstPatchVal;
			dstPatchVal.resize(tformDstPatch.size(), -1.f);
			for (int pp = 0; pp < tformDstPatch.size(); ++pp) {
				if (!isInside(target, tformDstPatch[pp].x, tformDstPatch[pp].y)) {
					oobMask[pp] = false;
					continue;
				}
				getInterpolatedPixel_single(gray_mat_[target], tformDstPatch[pp].x, tformDstPatch[pp].y, dstPatchVal[pp]);
			}

			for (int pp = 0; pp < sizePixels; ++pp) {

				// [occVal] true(255) = visible, false(0) = occluded
				bool occVal = (bool)(occMask_[source].at<uchar>((int)pixels[pp].y, (int)pixels[pp].x));

				// not occluded == visible
				if (occVal == true) {
					if (isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {


						dataCostMat.at<float>(pixels[pp]) = data_ternary_grad_rev1(source, target, oobMask, pixels[pp], tformedPixels[pp], tlPoint, wPatch, dstPatchVal, tformDstPatch);


					}
					else {	// cost = truncate_data, if it is out-of-bound
						dataCostMat.at<float>(pixels[pp]) = params_.truncate_data;
					}
				}
				else {
					dataCostMat.at<float>(pixels[pp]) = params_.lambda_occp;
				}
			}
		}
	}


	// Pairwise term
	cv::Mat pairwiseCostMat = cv::Mat(height_, width_, CV_64FC1, 0.f);
	{
		for (int bb = 0; bb < n_boundary; ++bb) {

			int idx_spx1 = boundaries_[view]->at(bb).getSegmentIndices().first;
			int idx_spx2 = boundaries_[view]->at(bb).getSegmentIndices().second;

			// calculate the pairwise cost between seg1 and seg2 
			cv::Mat state1 = segments_[view]->at(idx_spx1).getState();
			cv::Mat state2 = segments_[view]->at(idx_spx2).getState();

			std::vector<cv::Point2f> pixels1 = segments_[view]->at(idx_spx1).getPixelVector();
			std::vector<cv::Point2f> pixels2 = segments_[view]->at(idx_spx2).getPixelVector();
			std::vector<cv::Point2f> pixelsBoundary = boundaries_[view]->at(bb).getPixelVector();
			std::vector<float> weightBoundaryPixels = boundaries_[view]->at(bb).getPixelWeight();
			int sizePixel1 = (int)pixels1.size();
			int sizePixel2 = (int)pixels2.size();
			int sizeBoundary = (int)pixelsBoundary.size();

			float coPlanar_cost = 0.f;
			// calculating co-planar cost - l2 term
			{
				float energy1 = 0.f;
				std::vector<cv::Point2f> H1_pts1, H2_pts1;
				getTransformedPoints(pixels1, state1, H1_pts1);
				getTransformedPoints(pixels1, state2, H2_pts1);
				for (int pp = 0; pp < sizePixel1; ++pp) {
					energy1 += sqrtf((H1_pts1[pp].x - H2_pts1[pp].x) * (H1_pts1[pp].x - H2_pts1[pp].x) + (H1_pts1[pp].y - H2_pts1[pp].y) * (H1_pts1[pp].y - H2_pts1[pp].y));
				}

				float energy2 = 0.f;
				std::vector<cv::Point2f> H1_pts2, H2_pts2;
				getTransformedPoints(pixels2, state1, H1_pts2);
				getTransformedPoints(pixels2, state2, H2_pts2);
				for (int pp = 0; pp < sizePixel2; ++pp) {
					energy2 += sqrtf((H1_pts2[pp].x - H2_pts2[pp].x) * (H1_pts2[pp].x - H2_pts2[pp].x) + (H1_pts2[pp].y - H2_pts2[pp].y) * (H1_pts2[pp].y - H2_pts2[pp].y));
				}
				float bSpxWeight = boundaries_[view]->at(bb).getSpxWeight();
				coPlanar_cost = bSpxWeight * (energy1 + energy2) / (float)(sizePixel1 + sizePixel2);
				// coPlanar_cost = bSpxWeight * energy1 / (float)sizePixel + energy2 / (float)sizePixel2;
			}

			float cost = 0.f;
			// calculating hinge cost - l2 term
			{
				std::vector<cv::Point2f> H1_pts;
				std::vector<cv::Point2f> H2_pts;
				getTransformedPoints(pixelsBoundary, state1, H1_pts);
				getTransformedPoints(pixelsBoundary, state2, H2_pts);
				for (int pp = 0; pp < sizeBoundary; ++pp) {
					float weight = weightBoundaryPixels[pp];
					//float curr_energy = weight * std::min(abs(H1_pts[pp].x - H2_pts[pp].x) + abs(H1_pts[pp].y - H2_pts[pp].y), params_.truncate_pairwise);
					float disp = sqrtf((H1_pts[pp].x - H2_pts[pp].x) * (H1_pts[pp].x - H2_pts[pp].x) + (H1_pts[pp].y - H2_pts[pp].y) * (H1_pts[pp].y - H2_pts[pp].y));
					float min_energy = std::min(params_.truncate_pairwise, std::min(params_.weight_coPlanar * coPlanar_cost, params_.truncate_coPlanar + disp));
					float curr_energy = weight * min_energy;
					pairwiseCostMat.at<float>((int)pixelsBoundary[pp].y, (int)pixelsBoundary[pp].x) = curr_energy;
				}
			}
		}
	}



	// Consistency term
	cv::Mat consistencyCostMat = cv::Mat(height_, width_, CV_64FC1, 0.f);
	{
		for (int ss = 0; ss < n_segments; ++ss) {
			std::vector<cv::Point2f> pixels = segments_[view]->at(ss).getPixelVector();
			cv::Mat state = segments_[view]->at(ss).getState();

			std::vector<cv::Point2f> tformedPixels;
			getTransformedPoints(pixels, state, tformedPixels);
			int sizePixels = (int)pixels.size();

			for (int pp = 0; pp < sizePixels; ++pp) {

				if (isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {

					// [occVal] true(255) = visible, false(0) = occluded
					bool occVal_src = (bool)(occMask_[source].at<uchar>((int)pixels[pp].y, (int)pixels[pp].x));
					bool occVal_tgt = (bool)(occMask_[target].at<uchar>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f)));

					// not occluded
					if (occVal_src == true && occVal_tgt == true) {

						int spx_idx_tgt = (int)superpixelLabelMap_[target].at<unsigned short>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f));
						cv::Mat state_tgt = segments_[target]->at(spx_idx_tgt).getState();
						cv::Point2f pp_backprojected;
						getTransformedPoint(tformedPixels[pp], state_tgt, pp_backprojected);

						float dist = sqrtf((pixels[pp].x - pp_backprojected.x)*(pixels[pp].x - pp_backprojected.x) + (pixels[pp].y - pp_backprojected.y)*(pixels[pp].y - pp_backprojected.y));

						consistencyCostMat.at<float>((int)pixels[pp].y, (int)pixels[pp].x) = std::min(dist, params_.truncate_consistency);
					}
				}
			}
		}
	}

	// Symmetry term
	cv::Mat symmetryCostMat = cv::Mat(height_, width_, CV_64FC1, 0.f);
	{
		for (int hh = 0; hh < height_; ++hh){
			for (int ww = 0; ww < width_; ++ww){
				bool occupied = (bool)projectionMask_[view].at<uchar>(hh, ww);
				bool occVal = (bool)(occMask_[view].at<uchar>(hh, ww));

				if (occupied != occVal){
					symmetryCostMat.at<float>(hh, ww) = 1;
				}
			}
		}
	}


	// Displaying
	cv::Mat visData = cv::Mat(height_, width_, CV_8UC3, 0.f);
	cv::Mat visPair = cv::Mat(height_, width_, CV_8UC3, 0.f);
	cv::Mat visCons = cv::Mat(height_, width_, CV_8UC3, 0.f);
	cv::Mat visSymm = cv::Mat(height_, width_, CV_8UC3, 0.f);

	for (int hh = 0; hh < height_; ++hh){
		for (int ww = 0; ww < width_; ++ww){

			float dataVal = dataCostMat.at<float>(hh, ww);
			cv::Vec3b data_color;
			for (int32_t i = 0; i < 10; i++) {
				if (dataVal >= LC[i][0] && dataVal < LC[i][1]) {
					data_color = cv::Vec3b((uint8_t)LC[i][4], (uint8_t)LC[i][3], (uint8_t)LC[i][2]);
				}
			}
			visData.at<cv::Vec3b>(hh, ww) = data_color;

			float pairVal = pairwiseCostMat.at<float>(hh, ww);
			cv::Vec3b pair_color;
			for (int32_t i = 0; i < 10; i++) {
				if (pairVal >= LC_log[i][0] && pairVal < LC_log[i][1]) {
					pair_color = cv::Vec3b((uint8_t)LC_log[i][4], (uint8_t)LC_log[i][3], (uint8_t)LC_log[i][2]);
				}
			}
			visPair.at<cv::Vec3b>(hh, ww) = pair_color;

			float consVal = consistencyCostMat.at<float>(hh, ww) * 2.f;
			cv::Vec3b cons_color;
			for (int32_t i = 0; i < 10; i++) {
				if (consVal >= LC_log[i][0] && consVal < LC_log[i][1]) {
					cons_color = cv::Vec3b((uint8_t)LC_log[i][4], (uint8_t)LC_log[i][3], (uint8_t)LC_log[i][2]);
				}
			}
			visCons.at<cv::Vec3b>(hh, ww) = cons_color;

			float symmVal = symmetryCostMat.at<float>(hh, ww) * 16.f;
			cv::Vec3b symm_color;
			for (int32_t i = 0; i < 10; i++) {
				if (symmVal >= LC[i][0] && symmVal < LC[i][1]) {
					symm_color = cv::Vec3b((uint8_t)LC[i][4], (uint8_t)LC[i][3], (uint8_t)LC[i][2]);
				}
			}
			visSymm.at<cv::Vec3b>(hh, ww) = symm_color;
		}
	}

	if (init == true){
		if (view == currFrame){
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_data_init.png"), visData);
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_pair_init.png"), visPair);
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_cons_init.png"), visCons);
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_symm_init.png"), visSymm);
		}
		else{
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_data_init.png"), visData);
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_pair_init.png"), visPair);
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_cons_init.png"), visCons);
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_symm_init.png"), visSymm);
		}
	}
	else{
		if (view == currFrame){
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_data.png"), visData);
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_pair.png"), visPair);
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_cons.png"), visCons);
			cv::imwrite(std::string(params_.dirRes + "resE1_" + params_.prefix + "_symm.png"), visSymm);
		}
		else{
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_data.png"), visData);
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_pair.png"), visPair);
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_cons.png"), visCons);
			cv::imwrite(std::string(params_.dirRes + "resE2_" + params_.prefix + "_symm.png"), visSymm);
		}
	}
}

void OptModule::runQPBO_lae(const OptView view,
	const std::set<int>& setCenGrpSpx,
	const std::set<int>& setBndGrpSpx,
	const std::set<int>& setBnds,
	const std::vector<cv::Mat>& setStates){

	//log_file_ << "  run qpbo" << std::endl;

	//// sanity check
	//if ((int)setStates.size() > params_.n_random_states){
	//	std::cout << " Sanity Check Failed runQPBO_lae" << std::endl;
	//}
	//std::cout << setCenGrpSpx.size() << " " << setBndGrpSpx.size() << " " << setExpGrpSpx.size() << " " << setBnds.size() << " " << setStates.size() << std::endl;

	// a superpixel vector for optimization
	std::vector<int> vecOptSpx;
	std::map<int, int> mapSpxToIdx;
	for (auto iter_ss = setCenGrpSpx.begin(); iter_ss != setCenGrpSpx.end(); ++iter_ss) {
		int idx = (int)(vecOptSpx.size());
		vecOptSpx.push_back(*iter_ss);
		mapSpxToIdx.insert(std::pair<int, int>(*iter_ss, idx));
	}
	for (auto iter_ss = setBndGrpSpx.begin(); iter_ss != setBndGrpSpx.end(); ++iter_ss) {
		int idx = (int)(vecOptSpx.size());
		vecOptSpx.push_back(*iter_ss);
		mapSpxToIdx.insert(std::pair<int, int>(*iter_ss, idx));
	}

	// stores result of optimization
	int num_cenSpx = (int)(setCenGrpSpx.size());
	int num_bndSpx = (int)(setBndGrpSpx.size());
	int num_optSpx = (int)(vecOptSpx.size());
	int num_labels = (int)(setStates.size()) + 1;	// currentState + newState(from setStates)

	//// sanity check
	//if (num_cenSpx + num_bndSpx != num_optSpx) {
	//	std::cout << " ERROR OptModule.cpp : the number of superpixels calculated is wrong" << std::endl;
	//}


	float maxDataCost = 1000000.f;

	// current label
	std::vector<int> currLabel;
	currLabel.resize(num_optSpx, 0);
	std::vector<float> currUnary, nextUnary;
	currUnary.resize(num_optSpx, maxDataCost);
	nextUnary.resize(num_optSpx, maxDataCost);


	// calculate data - for the current states
	for (int ss = 0; ss < num_optSpx; ++ss) {
		int idx_spx = vecOptSpx[ss];
		cv::Mat currState = segments_[view]->at(idx_spx).getState();
		currUnary[ss] = dataCost(view, idx_spx, currState);
	}


	//log_file_ << "   qpbo start: " << std::endl;

	// for each label, run QPBO
	for (int ll = 1; ll < num_labels; ++ll){

		// QPBO definiion
		typedef float REAL;
		QPBO<REAL>* q;
		q = new QPBO<REAL>(num_optSpx, (int)(setBnds.size())); // max number of nodes & edges
		q->AddNode(num_optSpx); // add nodes

		// calculate Unary energy
		if (params_.fix_boundary_region){
			// calculate data - for the sampled states, for the Center Region Superpixels
			for (int ss = 0; ss < num_cenSpx; ++ss) {
				int idx_spx = vecOptSpx[ss];
				nextUnary[ss] = dataCost(view, idx_spx, setStates[ll - 1]);
			}
			// calculate data - for the sampled states, for the Boundary Region Superpixels
			for (int ss = num_cenSpx - 1; ss < num_optSpx; ++ss) {
				nextUnary[ss] = maxDataCost;
			}
		}
		else{
			for (int ss = 0; ss < num_optSpx; ++ss) {
				int idx_spx = vecOptSpx[ss];
				nextUnary[ss] = dataCost(view, idx_spx, setStates[ll - 1]);
			}
		}

		// add unary term		
		for (int ss = 0; ss < num_optSpx; ++ss){
			q->AddUnaryTerm(ss, currUnary[ss], nextUnary[ss]);
		}


		// calculate Pairwise energy &  add pairwise term
		for (auto iter_bb = setBnds.begin(); iter_bb != setBnds.end(); ++iter_bb) {
			int seg1 = boundaries_[view]->at(*iter_bb).getSegmentIndices().first;
			int seg2 = boundaries_[view]->at(*iter_bb).getSegmentIndices().second;
			auto iter_m1 = mapSpxToIdx.find(seg1);
			auto iter_m2 = mapSpxToIdx.find(seg2);

			// indices in vecOptSpx, calculate the pairwise cost between seg1 and seg2 using their index (seg1_idx, seg2_idx)
			int seg1_idx = iter_m1->second;
			int seg2_idx = iter_m2->second;

			// current state
			cv::Mat currState1, currState2;
			if (currLabel[seg1_idx] == 0){
				currState1 = segments_[view]->at(seg1).getState();
			}
			else{
				currState1 = setStates[currLabel[seg1_idx] - 1];
			}
			if (currLabel[seg2_idx] == 0){
				currState2 = segments_[view]->at(seg2).getState();
			}
			else{
				currState2 = setStates[currLabel[seg2_idx] - 1];
			}

			float pair_e00 = pairwiseCost_co_hi(view, seg1, currState1, seg2, currState2, *iter_bb);
			float pair_e01 = pairwiseCost_co_hi(view, seg1, currState1, seg2, setStates[ll - 1], *iter_bb);
			float pair_e10 = pairwiseCost_co_hi(view, seg1, setStates[ll - 1], seg2, currState2, *iter_bb);
			float pair_e11 = 0.f;
			q->AddPairwiseTerm(seg1_idx, seg2_idx, pair_e00, pair_e01, pair_e10, pair_e11);
		}

		q->Solve();

		//log_file_ << "      ----------------------- " << ll << std::endl;
		//q->ComputeWeakPersistencies();
		for (int ss = 0; ss < num_optSpx; ++ss){
			if (q->GetLabel(ss) == 1){
				currLabel[ss] = ll;
				currUnary[ss] = nextUnary[ss];
			}
			if (q->GetLabel(ss) == -1){
				//std::cout << " Warning: no label assigned in QPBO" << std::endl;
				//log_file_ << " Warning: no label assigned in QPBO" << std::endl;
			}
		}
		q->~QPBO();

	}
	//log_file_ << "   qpbo end" << std::endl;

	// update states
	if (params_.fix_boundary_region){
		for (int ss = 0; ss < num_cenSpx; ++ss) {

			int idx_spx = vecOptSpx[ss];
			int label = currLabel[ss];

			if (label == 0) {
				continue;
			}

			cv::Mat newState = setStates[label - 1];
			segments_[view]->at(idx_spx).setState(newState);

			updateFlowFromOptRes(view, idx_spx);
		}
	}
	else{
		for (int ss = 0; ss < num_optSpx; ++ss) {

			int idx_spx = vecOptSpx[ss];
			int label = currLabel[ss];

			if (label == 0) {
				continue;
			}

			cv::Mat newState = setStates[label - 1];
			segments_[view]->at(idx_spx).setState(newState);

			updateFlowFromOptRes(view, idx_spx);
		}
	}

	//delete[] smooth;
	//delete[] data;
}

void OptModule::optimize_O_lae(OptView view, int oo_iter) {

	int n_pixels = width_ * height_;
	int n_labels = 2;	// 0: occluded, 1: visible
	int *result = new int[n_pixels];   // stores result of optimization

	// first set up the array for data costs
	float *data = new float[n_pixels* n_labels];

	std::vector<float> dataCost;
	int n_indices = height_ * width_ * n_labels;

	dataCost = dataCost_occ(view);
	for (int idx = 0; idx < n_indices; ++idx) {
		data[idx] = dataCost[idx];
	}

	// next set up the array for smooth costs
	float *smooth = new float[n_labels*n_labels];
	for (int l2 = 0; l2 < n_labels; ++l2) {
		for (int l1 = 0; l1 < n_labels; ++l1) {
			smooth[l1 + l2*n_labels] = (l1 != l2 ? params_.lambda_occPott : 0);
		}
	}

	//// for debugging - visualize occ unary
	//{
	//	cv::Mat occImg = cv::Mat(height_, width_, CV_8UC1);
	//	for (int hh = 0; hh < height_; ++hh) {
	//		for (int ww = 0; ww < width_; ++ww) {
	//			int idx = (ww + width_* hh)*n_labels;

	//			if (data[idx] > data[idx + 1]){
	//				occImg.at<uchar>(hh, ww) = 255;
	//			}
	//			else{
	//				occImg.at<uchar>(hh, ww) = 0;
	//			}
	//		}
	//	}
	//	cv::imwrite(params_.dirRes + "res_" + params_.prefix + "_occImg_" + std::to_string(view) + "_" + std::to_string(oo_iter) + ".png", occImg);
	//}

	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(n_pixels, n_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// setting up 8 neighbor
		for (int hh = 0; hh < height_; ++hh) {
			for (int ww = 0; ww < width_; ++ww) {

				int curr_idx = ww + width_ * hh;

				// right
				if (ww + 1 <= width_ - 1) {
					int right_idx = (ww + 1) + width_ * hh;
					gc->setNeighbors(curr_idx, right_idx);
				}

				// below
				if (hh + 1 <= height_ - 1) {
					int below_idx = ww + width_ * (hh + 1);
					gc->setNeighbors(curr_idx, below_idx);

					// left below 
					if (ww - 1 >= 0) {
						int leftBelow_idx = (ww - 1) + width_ * (hh + 1);
						gc->setNeighbors(curr_idx, leftBelow_idx);
					}

					// right below
					if (ww + 1 <= width_ - 1) {
						int rightBelow_idx = (ww + 1) + width_ * (hh + 1);
						gc->setNeighbors(curr_idx, rightBelow_idx);
					}
				}
			}
		}

		for (int hh = 0; hh < height_; ++hh) {
			for (int ww = 0; ww < width_; ++ww) {
				bool occVal = (bool)(occMask_[view].at<uchar>(hh, ww));
				int pp_idx = ww + width_ * hh;
				if (occVal == true){	// visible
					gc->setLabel(pp_idx, 1);
				}
				else{
					gc->setLabel(pp_idx, 0);
				}
			}
		}

		//std::cout << "   Occ Optimization: " << std::endl;
		//std::cout << "     before: " << gc->compute_energy() << " = " << gc->giveDataEnergy() << " + " << gc->giveSmoothEnergy() << " + " << gc->giveLabelEnergy() << std::endl;		
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//std::cout << "     after : " << gc->compute_energy() << " = " << gc->giveDataEnergy() << " + " << gc->giveSmoothEnergy() << " + " << gc->giveLabelEnergy() << std::endl;	

		for (int ii = 0; ii < n_pixels; ++ii) {
			result[ii] = gc->whatLabel(ii);
		}

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	for (int hh = 0; hh < height_; ++hh) {
		for (int ww = 0; ww < width_; ++ww) {
			int pp_idx = ww + width_ * hh;

			if (result[pp_idx] == 0) {
				occMask_[view].at<uchar>(hh, ww) = (uchar)0;
			}
			else {
				occMask_[view].at<uchar>(hh, ww) = (uchar)255;
			}
		}
	}

	delete[] result;
	delete[] smooth;
	delete[] data;
}

void OptModule::optimize_H_qpbo_lae(const OptView view) {
	
#ifdef _OPENMP
	omp_set_num_threads(4);
	//std::cout << "max thread #: " << omp_get_max_threads() << std::endl;
#endif

	for (int gg = 0; gg < centerRegionGroup2hop_[view].size(); ++gg) {


#ifdef _OPENMP
#pragma omp parallel for 
#endif
		// for each set of superpixels, run the alpha expansion
		for (int rr = 0; rr < centerRegionGroup2hop_[view][gg].size(); ++rr) {

			// a set of superpixels - running alpha expansion on { CenterGroup U BoundaryGroup }, but the states of BoundaryGroup are fixed, collecting candidates labels from { expansionRegionGroup }
			std::set<int> setCenGrpSpx = centerRegionGroup2hop_[view][gg][rr];	// to be updated
			std::set<int> setBndGrpSpx = boundaryRegionGroup2hop_[view][gg][rr];	// as anchor points
			std::set<int> setExpGrpSpx = expansionRegionGroup2hop_[view][gg][rr];	// for collecting states
			std::set<int> setBnds = boundaryGroup2hop_[view][gg][rr];	// boundary indices for pairwise

			// propagation
			propagation_qpbo_lae(view, setCenGrpSpx, setBndGrpSpx, setExpGrpSpx, setBnds, params_.n_random_states);

			// randomization
			randomize_qpbo_lae(view, setCenGrpSpx, setBndGrpSpx, setExpGrpSpx, setBnds, params_.n_random_states);
		}
	}




	// a set of superpixels - running alpha expansion on { CenterGroup U BoundaryGroup }, but the states of BoundaryGroup are fixed, collecting candidates labels from { expansionRegionGroup }
	std::set<int> setCenGrpSpx = allSpxGroup_[view];	// to be updated
	std::set<int> setBndGrpSpx;	// as anchor points
	std::set<int> setExpGrpSpx = allSpxGroup_[view];	// for collecting states
	std::set<int> setBnds = boundaryGroup_[view];	// boundary indices for pairwise

	// propagation
	propagation_qpbo_lae(view, setCenGrpSpx, setBndGrpSpx, setExpGrpSpx, setBnds, params_.n_random_states_all);

	// randomization
	//randomize_qpbo_lae(view, setCenGrpSpx, setBndGrpSpx, setExpGrpSpx, setBnds, params_.n_random_states_all);

}

void OptModule::propagation_qpbo_lae(const OptView view, const std::set<int>& setCenGrpSpx, const std::set<int>& setBndGrpSpx, const std::set<int>& setExpGrpSpx, const std::set<int>& setBnds, const int n_samples) {

	if (setCenGrpSpx.size() == 0) {
		return;
	}

	{
		// randomly sampled superpixels from the expansion set
		bool add_noise = false;
		float noise_scale = 0.f;
		float noise_trans = 0.f;
		std::vector<cv::Mat> setStates = collectRandomNstates(view, setExpGrpSpx, n_samples, false, add_noise, noise_scale, noise_trans);

		//std::cout << "  LIST : ";
		//for (auto iter = setCenGrpSpx->begin(); iter != setCenGrpSpx->end(); ++iter){
		//	std::cout << *iter << " ";
		//}
		//std::cout << std::endl;

		// run graph-cut
		if (setStates.size() != 0) {
			runQPBO_lae(view, setCenGrpSpx, setBndGrpSpx, setBnds, setStates);
		}
	}

	// randomly sampled superpixels from the opposite view
	if (params_.bidirectional == true){

		std::vector<cv::Mat> vecStates_opp = collectRandomNstatesFromOpposite(view, setExpGrpSpx, n_samples, true);

		// run QPBO
		if (vecStates_opp.size() != 0) {
			runQPBO_lae(view, setCenGrpSpx, setBndGrpSpx, setBnds, vecStates_opp);
		}
	}
	
}

void OptModule::randomize_qpbo_lae(const OptView view, const std::set<int>& setCenGrpSpx, const std::set<int>& setBndGrpSpx, const std::set<int>& setExpGrpSpx, const std::set<int>& setBnds, const int n_samples, bool verbose) {

	if (setCenGrpSpx.size() == 0) {
		return;
	}

	// randomly sampled superpixels from the expansion set, adding noises
	{
		bool add_noise = true;
		float noise_scale = params_.size_perturb_scale;
		float noise_trans = params_.size_perturb_trans;

		while (noise_trans > 0.5f) {
			std::vector<cv::Mat> vecStates = collectRandomNstates(view, setExpGrpSpx, n_samples, false, add_noise, noise_scale, noise_trans);
			noise_scale = noise_scale / 2.f;
			noise_trans = noise_trans / 2.f;

			// run QPBO
			if (vecStates.size() != 0) {
				runQPBO_lae(view, setCenGrpSpx, setBndGrpSpx, setBnds, vecStates);
			}
		}
	}

	// randomly sampling state fusing states around
	{
		float ratio = 0.5f;
		while (ratio > 0.09f){
			std::vector<cv::Mat> vecStates = collectRandomNstatesFusingStateAround(view, setCenGrpSpx, n_samples, ratio);
			ratio -= 0.2f;

			// run QPBO
			if (vecStates.size() != 0) {
				runQPBO_lae(view, setCenGrpSpx, setBndGrpSpx, setBnds, vecStates);
			}
		}
	}
	
}

cv::Mat OptModule::GetHomographyStateFromFlowMap(OptView view, int index) {

	std::vector<cv::Point2f> srcPoints, dstPoints;

	srcPoints = segments_[view]->at(index).getPixelVector();
	int numPixels = (int)srcPoints.size();

	dstPoints.resize(numPixels);
	for (int pp = 0; pp < numPixels; ++pp) {
		float xx = srcPoints[pp].x;
		float yy = srcPoints[pp].y;
		float dx = flowX_[view].at<float>((int)yy, (int)xx);
		float dy = flowY_[view].at<float>((int)yy, (int)xx);
		dstPoints[pp] = cv::Point2f(xx + dx, yy + dy);
	}

	if (numPixels < 4){
		std::cout << " FATAL Error :: Number of pixels are less then 4" << std::endl;
	}

	//cv::Mat state = computeHomography_stable(srcPoints, dstPoints, true, 2);
	cv::Mat state = cv::findHomography(srcPoints, dstPoints, CV_RANSAC, 0.5);

	if (state.empty()){
		log_file_ << " warning from the FindHomoGraphy" << std::endl;
		state = cv::findHomography(srcPoints, dstPoints, 0);
		//state = computeHomography_stable(srcPoints, dstPoints, false);
	}
	state.convertTo(state, CV_32FC1);

	if (!isStateValid(view, index, state)){
		state = identityMat_;
	}

	return state;
}

void OptModule::updateFlowFromOptRes(const OptView view, const int idx_ss) {

	std::vector<cv::Point2f> srcPoints, dstPoints;
	srcPoints = segments_[view]->at(idx_ss).getPixelVector();
	cv::Mat state = segments_[view]->at(idx_ss).getState();
	getTransformedPoints(srcPoints, state, dstPoints);

	int size_pts = (int)srcPoints.size();

	// flow map refinement
	for (int pp = 0; pp < size_pts; ++pp){

		float dx = dstPoints[pp].x - srcPoints[pp].x;
		float dy = dstPoints[pp].y - srcPoints[pp].y;

		flowX_[view].at<float>((int)srcPoints[pp].y, (int)srcPoints[pp].x) = dx;
		flowY_[view].at<float>((int)srcPoints[pp].y, (int)srcPoints[pp].x) = dy;
	}
}

void OptModule::initializeState_lae(OptView view) {

	int n_segments = (int)(segments_[view]->size());

	for (int ss = 0; ss < n_segments; ++ss) {

		cv::Mat state = GetHomographyStateFromFlowMap(view, ss);	// maybe revive? reduce the number of function calls
		segments_[view]->at(ss).setState(state);
	}
}

bool OptModule::checkSimilarState_superpixelwise(const OptView view, const std::vector<cv::Mat>& setState, const cv::Mat& seedState, const int idx_ss) {

	if (seedState.empty()){
		return false;
	}
	const std::vector<cv::Point2f> vecCornerPts = segments_[view]->at(idx_ss).getVecCornerPts();

	std::vector<cv::Point2f> warpedPts1;
	std::vector<cv::Point2f> warpedPts2;
	bool isValidInput = true;
	float thres = 1.f;

	getTransformedPoints(vecCornerPts, seedState, warpedPts1);

	for (int ii = 0; ii < setState.size(); ++ii) {
		getTransformedPoints(vecCornerPts, setState[ii], warpedPts2);
		float err = sqrtf((warpedPts2[0].x - warpedPts1[0].x)*(warpedPts2[0].x - warpedPts1[0].x) + (warpedPts2[0].y - warpedPts1[0].y)*(warpedPts2[0].y - warpedPts1[0].y))
			+ sqrtf((warpedPts2[1].x - warpedPts1[1].x)*(warpedPts2[1].x - warpedPts1[1].x) + (warpedPts2[1].y - warpedPts1[1].y)*(warpedPts2[1].y - warpedPts1[1].y))
			+ sqrtf((warpedPts2[2].x - warpedPts1[2].x)*(warpedPts2[2].x - warpedPts1[2].x) + (warpedPts2[2].y - warpedPts1[2].y)*(warpedPts2[2].y - warpedPts1[2].y))
			+ sqrtf((warpedPts2[3].x - warpedPts1[3].x)*(warpedPts2[3].x - warpedPts1[3].x) + (warpedPts2[3].y - warpedPts1[3].y)*(warpedPts2[3].y - warpedPts1[3].y));

		if (err < thres) {
			isValidInput = false;
		}
	}

	return isValidInput;
}

bool OptModule::checkSimilarState(const std::vector<cv::Mat>& setState, const cv::Mat& seedState) {

	if (seedState.empty()){
		return false;
	}

	std::vector<cv::Point2f> warpedPts1;
	std::vector<cv::Point2f> warpedPts2;
	bool isValidInput = true;
	float thres = 1.f;

	getTransformedPoints(seedPts_, seedState, warpedPts1);

	for (int ii = 0; ii < setState.size(); ++ii) {
		getTransformedPoints(seedPts_, setState[ii], warpedPts2);
		float err = sqrtf((warpedPts2[0].x - warpedPts1[0].x)*(warpedPts2[0].x - warpedPts1[0].x) + (warpedPts2[0].y - warpedPts1[0].y)*(warpedPts2[0].y - warpedPts1[0].y))
			+ sqrtf((warpedPts2[1].x - warpedPts1[1].x)*(warpedPts2[1].x - warpedPts1[1].x) + (warpedPts2[1].y - warpedPts1[1].y)*(warpedPts2[1].y - warpedPts1[1].y))
			+ sqrtf((warpedPts2[2].x - warpedPts1[2].x)*(warpedPts2[2].x - warpedPts1[2].x) + (warpedPts2[2].y - warpedPts1[2].y)*(warpedPts2[2].y - warpedPts1[2].y))
			+ sqrtf((warpedPts2[3].x - warpedPts1[3].x)*(warpedPts2[3].x - warpedPts1[3].x) + (warpedPts2[3].y - warpedPts1[3].y)*(warpedPts2[3].y - warpedPts1[3].y));

		if (err < thres) {
			isValidInput = false;
		}
	}

	return isValidInput;
}

std::vector<cv::Mat> OptModule::collectRandomNstates(OptView view, const std::set<int>& set_in, int n_samples, bool refine, bool add_noise, float noise_scale, float noise_trans) {

	// sanity check
	if (set_in.empty()){
		log_file_ << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
		std::cout << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
	}

	n_samples = std::min((int)set_in.size(), n_samples);
	srand(time(NULL));
	auto engine = std::default_random_engine{};

	std::vector<int> vecIndices;
	std::vector<cv::Mat> resStates;
	for (auto iter = set_in.begin(); iter != set_in.end(); ++iter) {
		vecIndices.push_back(*iter);
	}

	std::shuffle(std::begin(vecIndices), std::end(vecIndices), engine);

	for (int nn = 0; nn < (int)vecIndices.size(); ++nn){
		int seedIdx = vecIndices[nn];
		cv::Mat seedState = segments_[view]->at(seedIdx).getState();

		if (add_noise) {
			cv::Mat perturbedState = getPerturbedState(view, seedIdx, seedState, noise_scale, noise_trans);
			seedState = perturbedState;
		}

		if (refine){
			cv::Mat refinedState = getRefinedState(view, seedIdx, seedState);
			seedState = refinedState;
		}

		// check whether there is a similar state
		if (checkSimilarState_superpixelwise(view, resStates, seedState, seedIdx) && isStateValid(view, seedIdx, seedState)) {
			resStates.push_back(seedState);
		}

		if ((int)resStates.size() >= n_samples){
			break;
		}
	}

	return resStates;
}

std::vector<cv::Mat> OptModule::collectRandomNstatesFromOpposite(OptView view, const std::set<int>& set_in, int n_samples, bool refine) {

	// sanity check
	if (set_in.empty()){
		log_file_ << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
		std::cout << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
	}

	OptView source = view;
	OptView target = OtherView(source);
	n_samples = std::min((int)set_in.size(), n_samples);
	srand(time(NULL));
	auto engine = std::default_random_engine{};

	std::vector<int> vecIndices;
	std::vector<cv::Mat> resStates;
	for (auto iter = set_in.begin(); iter != set_in.end(); ++iter) {
		vecIndices.push_back(*iter);
	}

	std::shuffle(std::begin(vecIndices), std::end(vecIndices), engine);

	for (int nn = 0; nn < (int)vecIndices.size(); ++nn){
		int seedIdx = vecIndices[nn];
		cv::Mat seedState = segments_[view]->at(seedIdx).getState();
		cv::Point2f cntP = segments_[view]->at(seedIdx).getCenterPoint();
		cv::Point2f dstP;
		getTransformedPoint(cntP, seedState, dstP);

		if (isInside(view, dstP.x, dstP.y)) {
			int spx_idx_tgt = (int)superpixelLabelMap_[target].at<unsigned short>((int)(dstP.y + 0.5f), (int)(dstP.x + 0.5f));
			cv::Mat state_tgt = segments_[target]->at(spx_idx_tgt).getState();
			cv::Mat seedState = state_tgt.inv();

			if (refine){
				cv::Mat refinedState = getRefinedState(view, seedIdx, seedState);
				seedState = refinedState;
			}

			// check whether there is a similar state
			if (checkSimilarState_superpixelwise(view, resStates, seedState, seedIdx) && isStateValid(view, seedIdx, seedState)) {
				resStates.push_back(seedState);
			}
		}
		if ((int)resStates.size() >= n_samples){
			break;
		}
	}

	return resStates;
}

std::vector<cv::Mat> OptModule::collectRandomNstatesFusingStateAround(OptView view, const std::set<int>& set_in, int n_samples, float ratio) {

	// sanity check
	if (set_in.empty()){
		log_file_ << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
		std::cout << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
	}

	n_samples = std::min((int)set_in.size(), n_samples);
	srand(time(NULL));
	auto engine = std::default_random_engine{};

	std::vector<int> vecIndices;
	std::vector<cv::Mat> resStates;
	for (auto iter = set_in.begin(); iter != set_in.end(); ++iter) {
		vecIndices.push_back(*iter);
	}

	std::shuffle(std::begin(vecIndices), std::end(vecIndices), engine);

	for (int nn = 0; nn < (int)vecIndices.size(); ++nn){
		int seedIdx = vecIndices[nn];
		cv::Mat seedState = getRandomHomographyFromAround(view, seedIdx, ratio);

		// check whether there is a similar state
		if (checkSimilarState_superpixelwise(view, resStates, seedState, seedIdx) && isStateValid(view, seedIdx, seedState)) {
			resStates.push_back(seedState);
		}

		if ((int)resStates.size() >= n_samples){
			break;
		}
	}

	return resStates;
}

std::vector<cv::Mat> OptModule::collectRandomNstatesUsingConsistency(OptView view, const std::set<int>& set_in, int n_samples) {

	// sanity check
	if (set_in.empty()){
		log_file_ << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
		std::cout << " Fatal Warning: index set is empty! - collectRandomNstates()" << std::endl;
	}

	n_samples = std::min((int)set_in.size(), n_samples);
	srand(time(NULL));
	auto engine = std::default_random_engine{};

	std::vector<int> vecIndices;
	std::vector<cv::Mat> resStates;
	for (auto iter = set_in.begin(); iter != set_in.end(); ++iter) {
		vecIndices.push_back(*iter);
	}

	// update consistency energy map
	updateConsistencyEnergyMap(view, set_in);

	// collect flow samples and estimate homography
	std::shuffle(std::begin(vecIndices), std::end(vecIndices), engine);

	float thres = 0.1f;

	for (int nn = 0; nn < (int)vecIndices.size(); ++nn){

		int seedIdx = vecIndices[nn];
		Segment* currSeg = &segments_[view]->at(seedIdx);

		std::vector<cv::Point2f> sampledSrc, sampledDst;

		int sizeNeighbor = currSeg->getNumberOfNeighbor();

		for (int nn = 0; nn < sizeNeighbor; ++nn){
			int neighborIndex = currSeg->getNeighbor(nn);
			Segment* neighborSeg = &segments_[view]->at(neighborIndex);
			std::vector<cv::Point2f> neighborPixels = neighborSeg->getPixelVector();
			int numberNeighborPixels = (int)neighborPixels.size();

			for (int pp = 0; pp < numberNeighborPixels; ++pp){

				if (consistencyEnergy_[view].at<float>(neighborPixels[pp]) < thres){
					sampledSrc.push_back(neighborPixels[pp]);
					float dx = flowX_[view].at<float>(neighborPixels[pp]);
					float dy = flowY_[view].at<float>(neighborPixels[pp]);
					cv::Point2f dst_p = cv::Point2f(neighborPixels[pp].x + dx, neighborPixels[pp].y + dy);
					sampledDst.push_back(dst_p);
				}
			}
		}

		if (sampledDst.size() < 6) {
			continue;
		}

		cv::Mat seedState;
		seedState = cv::findHomography(sampledSrc, sampledDst, CV_RANSAC, 0.5);
		//state = computeHomography_stable(sampledSrc, sampledDst, true, 0.5);

		if (seedState.empty()){
			//state = computeHomography_stable(sampledSrc, sampledDst, false);
			seedState = cv::findHomography(sampledSrc, sampledDst, 0);
		}

		seedState.convertTo(seedState, CV_32FC1);

		// check whether there is a similar state
		if (checkSimilarState_superpixelwise(view, resStates, seedState, seedIdx) && isStateValid(view, seedIdx, seedState)) {
			resStates.push_back(seedState);
		}

		if ((int)resStates.size() >= n_samples){
			break;
		}
	}

	return resStates;
}

cv::Mat OptModule::computeHomography_stable(const std::vector<cv::Point2f>& srcP, const std::vector<cv::Point2f>& dstP, bool bRansac, float thres){

	// sanity check
	if (srcP.size() != dstP.size()){
		std::cout << " Sanity Check Failure, homography calculation " << srcP.size() << " != " << dstP.size() << std::endl;
	}
	int n_pts = (int)(srcP.size());
	std::vector<cv::Point2f> srcP_norm, dstP_norm;
	srcP_norm.resize(n_pts);
	dstP_norm.resize(n_pts);


	cv::Mat matNormSrc = cv::Mat(3, 3, CV_32FC1);
	cv::Mat matNormDst = cv::Mat(3, 3, CV_32FC1);
	{
		float mean_x = 0.f;
		float mean_y = 0.f;

		for (int ii = 0; ii < n_pts; ++ii){
			mean_x += srcP[ii].x;
			mean_y += srcP[ii].y;
		}

		mean_x = mean_x / ((float)n_pts);
		mean_y = mean_y / ((float)n_pts);

		float dist = 0.f;
		for (int ii = 0; ii < n_pts; ++ii){
			dist += sqrtf((srcP[ii].x - mean_x) * (srcP[ii].x - mean_x) + (srcP[ii].y - mean_y) * (srcP[ii].y - mean_y));
		}
		float scale = (float)(2 * n_pts / dist);

		matNormSrc.ptr<float>(0)[0] = scale;
		matNormSrc.ptr<float>(0)[2] = -scale * mean_x;
		matNormSrc.ptr<float>(1)[1] = scale;
		matNormSrc.ptr<float>(1)[2] = -scale * mean_y;
		matNormSrc.ptr<float>(2)[2] = 1;
	}

	{
		float mean_x = 0.f;
		float mean_y = 0.f;

		for (int ii = 0; ii < n_pts; ++ii){
			mean_x += dstP[ii].x;
			mean_y += dstP[ii].y;
		}

		mean_x = mean_x / ((float)n_pts);
		mean_y = mean_y / ((float)n_pts);

		float dist = 0.f;
		for (int ii = 0; ii < n_pts; ++ii){
			dist += sqrtf((dstP[ii].x - mean_x) * (dstP[ii].x - mean_x) + (dstP[ii].y - mean_y) * (dstP[ii].y - mean_y));
		}
		float scale = (float)(2 * n_pts / dist);

		matNormDst.ptr<float>(0)[0] = scale;
		matNormDst.ptr<float>(0)[2] = -scale * mean_x;
		matNormDst.ptr<float>(1)[1] = scale;
		matNormDst.ptr<float>(1)[2] = -scale * mean_y;
		matNormDst.ptr<float>(2)[2] = 1;
	}

	getTransformedPoints(srcP, matNormSrc, srcP_norm);
	getTransformedPoints(dstP, matNormDst, dstP_norm);

	cv::Mat state;
	if (bRansac){
		state = cv::findHomography(srcP_norm, dstP_norm, CV_RANSAC, thres);
	}
	else{
		state = cv::findHomography(srcP_norm, dstP_norm, 0);
	}

	if (state.empty()){
		if (bRansac){
			state = cv::findHomography(srcP, dstP, CV_RANSAC, thres);
			if (state.empty()){
				state = cv::findHomography(srcP, dstP, 0);
			}
		}
		else{
			state = cv::findHomography(srcP, dstP, 0);
		}
		state.convertTo(state, CV_32FC1);

		if (!state.empty()){
			float norm_const = state.ptr<float>(2)[2];
			for (int ii = 0; ii < 3; ++ii){
				for (int jj = 0; jj < 3; ++jj){
					state.ptr<float>(ii)[jj] = state.ptr<float>(ii)[jj] / norm_const;
				}
			}
		}
		return state;
	}
	else{
		state.convertTo(state, CV_32FC1);
		state = matNormDst.inv() * state * matNormSrc;

		if (!state.empty()){
			float norm_const = state.ptr<float>(2)[2];
			for (int ii = 0; ii < 3; ++ii){
				for (int jj = 0; jj < 3; ++jj){
					state.ptr<float>(ii)[jj] = state.ptr<float>(ii)[jj] / norm_const;
				}
			}
		}
		return state;
	}
	return state;
}

cv::Mat OptModule::getRandomHomographyFromAround(OptView view, int index, float ratio) {

	std::vector<cv::Point2f> srcPoints, sampledSrc, sampledDst;

	Segment* currSeg = &segments_[view]->at(index);
	srcPoints = currSeg->getPixelVector();

	int sizeNeighbor = currSeg->getNumberOfNeighbor();
	int maxSample = 20;
	int sizeSrcSample = (int)(std::min((int)srcPoints.size(), maxSample) * ratio);
	int sizeNeighborSample = (int)(std::min((int)srcPoints.size(), maxSample) * (1.f - ratio) / (float)(sizeNeighbor));
	sampledSrc.resize(sizeSrcSample + sizeNeighborSample*sizeNeighbor);
	sampledDst.resize(sizeSrcSample + sizeNeighborSample*sizeNeighbor);

	if (sampledSrc.size() < 6){
		return cv::Mat();
	}

	// Sampling Points
	srand(time(NULL));

	for (int pp = 0; pp < sizeSrcSample; ++pp){
		int randomIndex = rand() % srcPoints.size();
		sampledSrc[pp] = srcPoints[randomIndex];
		float xx = srcPoints[randomIndex].x;
		float yy = srcPoints[randomIndex].y;
		float dx = flowX_[view].at<float>((int)yy, (int)xx);
		float dy = flowY_[view].at<float>((int)yy, (int)xx);
		sampledDst[pp] = cv::Point2f(xx + dx, yy + dy);
	}

	for (int nn = 0; nn < sizeNeighbor; ++nn){
		int neighborIndex = currSeg->getNeighbor(nn);
		Segment* neighborSeg = &segments_[view]->at(neighborIndex);
		std::vector<cv::Point2f> neighborPixels = neighborSeg->getPixelVector();
		int numberNeighborPixels = (int)neighborPixels.size();

		for (int pp = 0; pp < sizeNeighborSample; ++pp){

			int randomIndex = rand() % numberNeighborPixels;
			sampledSrc[pp + sizeNeighborSample*nn + sizeSrcSample] = neighborPixels[randomIndex];
			float xx = neighborPixels[randomIndex].x;
			float yy = neighborPixels[randomIndex].y;
			float dx = flowX_[view].at<float>((int)yy, (int)xx);
			float dy = flowY_[view].at<float>((int)yy, (int)xx);
			sampledDst[pp + sizeNeighborSample*nn + sizeSrcSample] = cv::Point2f(xx + dx, yy + dy);

		}
	}

	cv::Mat state;

	for (int nn = 1; nn < 4; ++nn){
		state = cv::findHomography(sampledSrc, sampledDst, CV_RANSAC, nn - 0.5);
		//state = computeHomography_stable(sampledSrc, sampledDst, true, nn);
		if (!state.empty()){
			break;
		}
	}
	if (state.empty()){
		//state = computeHomography_stable(sampledSrc, sampledDst, false);
		state = cv::findHomography(sampledSrc, sampledDst, 0);
	}

	state.convertTo(state, CV_32FC1);
	return state;
}

cv::Mat OptModule::getPerturbedState(OptView view, int idx_ss, const cv::Mat& state_in, float noise_scale, float noise_trans) {

	srand(time(NULL));

	// get corner points
	const std::vector<cv::Point2f> vecCornerPts = segments_[view]->at(idx_ss).getVecCornerPts();
	std::vector<cv::Point2f> vecCornerPts_tf;
	getTransformedPoints(vecCornerPts, state_in, vecCornerPts_tf);

	float tx = 0.0001f*(rand() % 10000 - 5000.f) * noise_trans;
	float ty = 0.0001f*(rand() % 10000 - 5000.f) * noise_trans;

	for (int ii = 0; ii < vecCornerPts.size(); ++ii){
		float dx = 0.0001f*(rand() % 10000 - 5000.f) * noise_scale;
		float dy = 0.0001f*(rand() % 10000 - 5000.f) * noise_scale;
		vecCornerPts_tf[ii].x += dx + tx;
		vecCornerPts_tf[ii].y += dy + ty;
	}

	//cv::Mat noiseMat = computeHomography_stable(vecCornerPts, vecCornerPts_tf, false);
	cv::Mat noiseMat = cv::findHomography(vecCornerPts, vecCornerPts_tf, 0);

	if (noiseMat.empty()){
		log_file_ << " Fatal error: noiseMat is incorrupted " << std::endl;
		std::cout << " Fatal error: noiseMat is incorrupted " << std::endl;
	}
	noiseMat.convertTo(noiseMat, CV_32FC1);

	return noiseMat;
}

cv::Mat OptModule::getRefinedState(OptView view, const int idx_ss, const cv::Mat& state_in) {

	float imageData_sum = 0.f;
	float stepSizeForGrad = 1.0f;
	float stepSizeForMoving = 0.00005f;

	const std::vector<cv::Point2f> vecCornerPts = segments_[view]->at(idx_ss).getVecCornerPts();
	std::vector<cv::Point2f> vecCornerPts_tf;
	getTransformedPoints(vecCornerPts, state_in, vecCornerPts_tf);

	std::vector<float> grad_gg;
	grad_gg.resize(8);
	float dataCost_org;

	dataCost_org = dataCost(view, idx_ss, state_in, false);
	float pairwiseCost_org = 0.f;

	Segment* currSeg = &segments_[view]->at(idx_ss);
	int n_neighbors = currSeg->getNumberOfNeighbor();
	for (int nn = 0; nn < n_neighbors; ++nn){
		int nn_idx = currSeg->getNeighbor(nn);
		int b_idx = currSeg->getBoundary(nn);
		cv::Mat nn_state = segments_[view]->at(nn_idx).getState();
		pairwiseCost_org += pairwiseCost_co_hi(view, idx_ss, state_in, nn_idx, nn_state, b_idx);
	}

	float allEnergy_org = dataCost_org + pairwiseCost_org;

	// calculate numerical gradient
	for (int gg = 0; gg < 8; ++gg){

		// get new homography by adding a step.
		std::vector<cv::Point2f> vecCornerPts_tf_new = vecCornerPts_tf;

		if (gg == 0){
			vecCornerPts_tf_new[0].x += stepSizeForGrad;
		}
		else if (gg == 1){
			vecCornerPts_tf_new[0].y += stepSizeForGrad;
		}
		else if (gg == 2){
			vecCornerPts_tf_new[1].x += stepSizeForGrad;
		}
		else if (gg == 3){
			vecCornerPts_tf_new[1].y += stepSizeForGrad;
		}
		else if (gg == 4){
			vecCornerPts_tf_new[2].x += stepSizeForGrad;
		}
		else if (gg == 5){
			vecCornerPts_tf_new[2].y += stepSizeForGrad;
		}
		else if (gg == 6){
			vecCornerPts_tf_new[3].x += stepSizeForGrad;
		}
		else if (gg == 7){
			vecCornerPts_tf_new[3].y += stepSizeForGrad;
		}

		cv::Mat stateNew = cv::findHomography(vecCornerPts, vecCornerPts_tf_new, 0);
		//cv::Mat stateNew = computeHomography_stable(vecCornerPts, vecCornerPts_tf_new, false);

		stateNew.convertTo(stateNew, CV_32FC1);

		if (!isStateValid(view, idx_ss, stateNew)){
			return cv::Mat();
		}

		float dataCost_step;

		dataCost_step = dataCost(view, idx_ss, stateNew, false);

		float pairwiseCost_step = 0.f;
		for (int nn = 0; nn < n_neighbors; ++nn){
			int nn_idx = currSeg->getNeighbor(nn);
			int b_idx = currSeg->getBoundary(nn);
			cv::Mat nn_state = segments_[view]->at(nn_idx).getState();
			pairwiseCost_step += pairwiseCost_co_hi(view, idx_ss, stateNew, nn_idx, nn_state, b_idx);
		}

		grad_gg[gg] = ((dataCost_step + pairwiseCost_step) - allEnergy_org) / stepSizeForGrad;

	}

	for (int ii = 0; ii < 4; ++ii){
		vecCornerPts_tf[ii].x -= grad_gg[2 * ii] * stepSizeForMoving;
		vecCornerPts_tf[ii].y -= grad_gg[2 * ii + 1] * stepSizeForMoving;
	}


	cv::Mat refinedMat = cv::findHomography(vecCornerPts, vecCornerPts_tf, 0);
	//cv::Mat refinedMat = computeHomography_stable(vecCornerPts, vecCornerPts_tf, false);

	if (refinedMat.empty()){
		log_file_ << " Fatal error: noiseMat is incorrupted " << std::endl;
		std::cout << " Fatal error: noiseMat is incorrupted " << std::endl;
	}
	refinedMat.convertTo(refinedMat, CV_32FC1);

	return refinedMat;
}

inline int OptModule::hammingDistanceTernary(double in1, double in2) const {

	int dist = 0;
	while (in1 > 0.9 || in2 > 0.9) {
		int val1 = ((int)in1) % 3;
		int val2 = ((int)in2) % 3;
		if (val1 != val2) {
			dist++;
		}
		in1 = (in1 - val1) / 3;
		in2 = (in2 - val2) / 3;

	}
	return dist;

}

inline void OptModule::hammingDistanceTernary_cont(std::vector<float> ter1, std::vector<float> ter2, float& dist, int& n_oob){

	n_oob = 0;
	dist = 0.f;

	for (int ii = 0; ii < ternaryPatchSize_; ++ii){
		if (ter1[ii] == ternaryOobCode_ || ter2[ii] == ternaryOobCode_){
			n_oob++;
			continue;
		}

		// Geman and McClure
		float val = ter1[ii] - ter2[ii];
		dist += val * val / (0.1f + val * val);
	}

	return;
}

inline float OptModule::ternary_sigmoid(float val) const{

	//return val / sqrtf(0.81f + val*val);
	return val / (0.45f + fabsf(val));
}

double OptModule::getTernaryValfromPatch(OptView view, const cv::Point2f& srcP, const cv::Point2i& tlPoint, int wPatch, const std::vector<float>& dstPatchVal, const std::vector<cv::Point2f>& tformDstPatch, int ternarySize) const {

	double valCensus = 0;
	float offset = 1.25f;
	int xCenter = (int)(srcP.x - (tlPoint.x - ternarySize));
	int yCenter = (int)(srcP.y - (tlPoint.y - ternarySize));
	int centerIdx = xCenter + wPatch * yCenter;

	if (!isInside(view, tformDstPatch[centerIdx].x, tformDstPatch[centerIdx].y)) {
		valCensus = -1;
	}
	else {
		float val_dst_c = dstPatchVal[centerIdx];
		for (int offsetY = -ternarySize; offsetY <= ternarySize; ++offsetY)
		for (int offsetX = -ternarySize; offsetX <= ternarySize; ++offsetX) {

			valCensus = valCensus * 3;

			int pIdx = (xCenter + offsetX) + wPatch * (yCenter + offsetY);

			if (isInside(view, tformDstPatch[pIdx].x, tformDstPatch[pIdx].y)) {

				float val_t = dstPatchVal[pIdx];
				double census_in = 0.0;
				if (val_t >= val_dst_c + offset) {
					census_in = 2.0;
				}
				else if (val_t <= val_dst_c - offset) {
					census_in = 0.0;
				}
				else {
					census_in = 1.0;
				}
				valCensus += census_in;
			}
		}
	}

	return valCensus;

}

std::vector<float> OptModule::getTernaryValfromPatch_cont(OptView view, const cv::Point2f& srcP, const cv::Point2i& tlPoint, int wPatch, const std::vector<float>& dstPatchVal, const std::vector<cv::Point2f>& tformDstPatch) const {

	std::vector<float> ternaryVal;
	ternaryVal.resize((2 * ternaryHalfWidth_ + 1)*(2 * ternaryHalfWidth_ + 1), ternaryOobCode_);

	int xCenter = (int)(srcP.x - (tlPoint.x - ternaryHalfWidth_));
	int yCenter = (int)(srcP.y - (tlPoint.y - ternaryHalfWidth_));
	int centerIdx = xCenter + wPatch * yCenter;

	if (!isInside(view, tformDstPatch[centerIdx].x, tformDstPatch[centerIdx].y)) {
		return ternaryVal;
	}
	else {
		float val_dst_c = dstPatchVal[centerIdx];
		int idx_cc = 0;

		for (int offsetY = -ternaryHalfWidth_; offsetY <= ternaryHalfWidth_; ++offsetY)
		for (int offsetX = -ternaryHalfWidth_; offsetX <= ternaryHalfWidth_; ++offsetX) {

			int pIdx = (xCenter + offsetX) + wPatch * (yCenter + offsetY);

			if (isInside(view, tformDstPatch[pIdx].x, tformDstPatch[pIdx].y)) {
				float val = dstPatchVal[pIdx] - val_dst_c;
				ternaryVal[idx_cc] = ternary_sigmoid(val);
			}
			else{
				ternaryVal[idx_cc] = ternaryOobCode_;
			}
			idx_cc++;
		}
	}

	return ternaryVal;

}

float OptModule::dataCost(OptView view, const int idx_spx, const cv::Mat& state, bool verbose) {

	std::vector<cv::Point2f> pixels = segments_[view]->at(idx_spx).getPixelVector();
	cv::Point2i tlPt = segments_[view]->at(idx_spx).getCensusPatchTL();
	cv::Point2i brPt = segments_[view]->at(idx_spx).getCensusPatchBR();

	// Data term
	float dataTermCost = dataTerm_cont(view, pixels, state, tlPt, brPt);

	// Consistency term
	float consistTermCost = consistencyTerm(view, pixels, state);

	return (dataTermCost + consistTermCost);

}

float OptModule::dataTerm_cont(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state, const cv::Point2i& tlPoint, const cv::Point2i& brPoint) {

	OptView source = view;
	OptView target = OtherView(source);

	// perspectiveCensus, create patch 
	int wPatch = brPoint.x - tlPoint.x + 2 * ternaryHalfWidth_ + 1;
	int hPatch = brPoint.y - tlPoint.y + 2 * ternaryHalfWidth_ + 1;
	std::vector<cv::Point2f> dstPatch, tformDstPatch;
	dstPatch.resize(wPatch*hPatch);
	tformDstPatch.resize(wPatch*hPatch);
	std::vector<bool> oobMask;
	oobMask.resize(wPatch*hPatch, true);	// true: inside of the image, false: out-of-bound pixels
	int count_pp = 0;
	for (int hh = tlPoint.y - ternaryHalfWidth_; hh <= brPoint.y + ternaryHalfWidth_; ++hh) {
		for (int ww = tlPoint.x - ternaryHalfWidth_; ww <= brPoint.x + ternaryHalfWidth_; ++ww) {
			dstPatch[count_pp] = cv::Point2f((float)ww, (float)hh);
			count_pp++;
		}
	}

	// get transformed pts
	getTransformedPoints(dstPatch, state, tformDstPatch);

	// color interpolation
	std::vector<float> dstPatchVal;
	dstPatchVal.resize(tformDstPatch.size(), -1.f);
	for (int pp = 0; pp < tformDstPatch.size(); ++pp) {
		if (!isInside(target, tformDstPatch[pp].x, tformDstPatch[pp].y)) {
			oobMask[pp] = false;
			continue;
		}
		getInterpolatedPixel_single(gray_mat_[target], tformDstPatch[pp].x, tformDstPatch[pp].y, dstPatchVal[pp]);
	}

	float cost = 0.f;
	std::vector<cv::Point2f> tformedPixels;
	getTransformedPoints(pixels, state, tformedPixels);
	int sizePixels = (int)(pixels.size());
	for (int pp = 0; pp < sizePixels; ++pp) {

		// [occVal] true(255) = visible, false(0) = occluded
		bool occVal = (bool)(occMask_[source].at<uchar>((int)pixels[pp].y, (int)pixels[pp].x));

		// not occluded == visible
		if (occVal == true) {
			if (isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {
				//cost += data_ternary_grad_rev1(source, target, oobMask, pixels[pp], tformedPixels[pp], tlPoint, wPatch, dstPatchVal, tformDstPatch);
				//cost += data_ternary_grad(source, target, pixels[pp], tformedPixels[pp], tlPoint, wPatch, dstPatchVal, tformDstPatch);
				//cost += data_intensity_grad(source, target, pixels[pp], tformedPixels[pp]);	

				int idx_pp = (int)(pixels[pp].x + (int)pixels[pp].y * width_);

				// ternary
				std::vector<float> ternary1 = ternary_mat_cont[source][idx_pp];
				std::vector<float> ternary2;
				float hammDist_pre = 0.f;
				int n_oob = 0;

				{
					ternary2.resize(ternaryPatchSize_, ternaryOobCode_);

					int xCenter = (int)(pixels[pp].x - (tlPoint.x - ternaryHalfWidth_));
					int yCenter = (int)(pixels[pp].y - (tlPoint.y - ternaryHalfWidth_));

					float val_dst_c = dstPatchVal[xCenter + wPatch * yCenter];
					int idx_cc = 0;

					for (int offsetY = -ternaryHalfWidth_; offsetY <= ternaryHalfWidth_; ++offsetY){
						for (int offsetX = -ternaryHalfWidth_; offsetX <= ternaryHalfWidth_; ++offsetX) {

							if (ternary1[idx_cc] == ternaryOobCode_){
								++n_oob;
								++idx_cc;
								continue;
							}
							int pIdx = (xCenter + offsetX) + wPatch * (yCenter + offsetY);
							if (!oobMask[pIdx]) {
								++n_oob;
								++idx_cc;
								continue;
							}

							// hammingDist(ter1[idx_cc], ter2[idx_cc]) - Geman and McClure
							float val = ternary1[idx_cc] - ternary_sigmoid(dstPatchVal[pIdx] - val_dst_c);
							hammDist_pre += val * val / (0.1f + val * val);
							++idx_cc;
						}
					}
				}

				if (n_oob > (int)(ternaryPatchSize_ * 0.75f)){
					cost += params_.truncate_data;
					continue;
				}

				float perpixData = params_.alpha * std::min(hammDist_pre * (float)ternaryPatchSize_ / (float)(ternaryPatchSize_ - n_oob), params_.truncate_ternary);

				// gradient
				float gradtX = 0.f, gradtY = 0.f;
				getInterpolatedPixel_single_float(gradX_mat_[target], tformedPixels[pp].x, tformedPixels[pp].y, gradtX);
				getInterpolatedPixel_single_float(gradY_mat_[target], tformedPixels[pp].x, tformedPixels[pp].y, gradtY);
				float diffX = gradX_mat_[source].at<float>(pixels[pp]) - gradtX;
				float diffY = gradY_mat_[source].at<float>(pixels[pp]) - gradtY;
				float absDiff = std::min(fabsf(diffX) + fabsf(diffY), params_.truncate_grad);
				perpixData += (1.f - params_.alpha)*absDiff;

				cost += 5.08f * logf(1.f + perpixData*perpixData / 8.f);	// 5.08 is for setting offset, the lorentzian function pass (20,20)

			}
			else {	// cost = truncate_data, if it is out-of-bound
				cost += params_.truncate_data;
			}
		}
		else {
			cost += params_.lambda_occp;
		}
	}

	return cost;
}

float OptModule::data_ternary_grad(OptView source, OptView target, const cv::Point2f& pixel, cv::Point2f& tformPixel, const cv::Point2i& tlPoint, int wPatch, std::vector<float>& dstPatchVal, std::vector<cv::Point2f>& tformDstPatch){

	float perpixData = 0.f;

	int idx_pp = (int)pixel.x + (int)pixel.y * width_;

	// ternary
	std::vector<float> ternary1 = ternary_mat_cont[source][idx_pp];
	std::vector<float> ternary2 = getTernaryValfromPatch_cont(target, pixel, tlPoint, wPatch, dstPatchVal, tformDstPatch);

	float hammDist_pre = 0.f;
	int n_oob = 0;
	hammingDistanceTernary_cont(ternary1, ternary2, hammDist_pre, n_oob);

	if (n_oob > (int)(ternaryPatchSize_ * 0.75f)){
		return params_.truncate_data;
	}

	float hammDist = hammDist_pre * (float)ternaryPatchSize_ / (float)(ternaryPatchSize_ - n_oob);
	float truncHammingDistance = std::min(hammDist, params_.truncate_ternary);
	perpixData += params_.alpha*truncHammingDistance;

	// gradient
	float gradtX = 0.f, gradtY = 0.f;
	getInterpolatedPixel_single_float(gradX_mat_[target], tformPixel.x, tformPixel.y, gradtX);
	getInterpolatedPixel_single_float(gradY_mat_[target], tformPixel.x, tformPixel.y, gradtY);
	float diffX = gradX_mat_[source].at<float>(pixel) -gradtX;
	float diffY = gradY_mat_[source].at<float>(pixel) -gradtY;
	float absDiff = std::min(fabsf(diffX) + fabsf(diffY), params_.truncate_grad);
	perpixData += (1.f - params_.alpha)*absDiff;

	float lorentzian = 5.08f * logf(1 + perpixData*perpixData / 2.f / 2.f / 2.f);	// 5.08 is for setting offset, the lorentzian function pass (20,20)

	return lorentzian;
}

float OptModule::data_ternary_grad_rev1(OptView source, OptView target, std::vector<bool> oobMask, const cv::Point2f& pixel, const cv::Point2f& tformPixel, const cv::Point2i& tlPoint, int wPatch, std::vector<float>& dstPatchVal, std::vector<cv::Point2f>& tformDstPatch){

	float perpixData = 0.f;
	int idx_pp = (int)pixel.x + (int)pixel.y * width_;

	// ternary
	std::vector<float> ternary1 = ternary_mat_cont[source][idx_pp];
	std::vector<float> ternary2;
	float hammDist_pre = 0.f;
	int n_oob = 0;

	{
		ternary2.resize(ternaryPatchSize_, ternaryOobCode_);

		int xCenter = (int)(pixel.x - (tlPoint.x - ternaryHalfWidth_));
		int yCenter = (int)(pixel.y - (tlPoint.y - ternaryHalfWidth_));
		int centerIdx = xCenter + wPatch * yCenter;

		float val_dst_c = dstPatchVal[centerIdx];
		int idx_cc = 0;

		for (int offsetY = -ternaryHalfWidth_; offsetY <= ternaryHalfWidth_; ++offsetY){
			for (int offsetX = -ternaryHalfWidth_; offsetX <= ternaryHalfWidth_; ++offsetX) {

				if (ternary1[idx_cc] == ternaryOobCode_){
					n_oob++;
					idx_cc++;
					continue;
				}
				int pIdx = (xCenter + offsetX) + wPatch * (yCenter + offsetY);
				if (!oobMask[pIdx]) {
					n_oob++;
					idx_cc++;
					continue;
				}

				// hammingDist(ter1[idx_cc], ter2[idx_cc]) - Geman and McClure
				float val = ternary1[idx_cc] - ternary_sigmoid(dstPatchVal[pIdx] - val_dst_c);
				hammDist_pre += val * val / (0.1f + val * val);
				idx_cc++;
			}
		}
	}

	if (n_oob > (int)(ternaryPatchSize_ * 0.75f)){
		return params_.truncate_data;
	}

	float hammDist = hammDist_pre * (float)ternaryPatchSize_ / (float)(ternaryPatchSize_ - n_oob);
	float truncHammingDistance = std::min(hammDist, params_.truncate_ternary);
	perpixData = params_.alpha * truncHammingDistance;

	// gradient
	float gradtX = 0.f, gradtY = 0.f;
	getInterpolatedPixel_single_float(gradX_mat_[target], tformPixel.x, tformPixel.y, gradtX);
	getInterpolatedPixel_single_float(gradY_mat_[target], tformPixel.x, tformPixel.y, gradtY);
	float diffX = gradX_mat_[source].at<float>(pixel) -gradtX;
	float diffY = gradY_mat_[source].at<float>(pixel) -gradtY;
	float absDiff = std::min(fabsf(diffX) + fabsf(diffY), params_.truncate_grad);
	perpixData += (1.f - params_.alpha)*absDiff;

	float lorentzian = 5.08f * logf(1.f + perpixData*perpixData / 8.f);	// 5.08 is for setting offset, the lorentzian function pass (20,20)

	return lorentzian;
}

float OptModule::data_intensity_grad(OptView source, OptView target, const cv::Point2f& pixel, const cv::Point2f& tformPixel){

	float perpixData = 0.f;

	// Intensity
	float srcVal = (float)(gray_mat_[source].at<uchar>(pixel.y, pixel.x));
	float dstVal = 0.f;
	getInterpolatedPixel_single(gray_mat_[target], tformPixel.x, tformPixel.y, dstVal);
	float truncDistance = std::min(fabsf(srcVal - dstVal), params_.truncate_ternary);
	perpixData += params_.alpha*truncDistance;

	// gradient
	float gradtX = 0.f, gradtY = 0.f;
	getInterpolatedPixel_single_float(gradX_mat_[target], tformPixel.x, tformPixel.y, gradtX);
	getInterpolatedPixel_single_float(gradY_mat_[target], tformPixel.x, tformPixel.y, gradtY);
	float diffX = gradX_mat_[source].at<float>(pixel) -gradtX;
	float diffY = gradY_mat_[source].at<float>(pixel) -gradtY;
	float absDiff = std::min(fabsf(diffX) + fabsf(diffY), params_.truncate_grad);
	perpixData += (1.f - params_.alpha)*absDiff;

	float lorentzian = 5.08f * logf(1 + perpixData*perpixData / 2.f / 2.f / 2.f);	// 5.08 is for setting offset, the lorentzian function pass (20,20)

	return lorentzian;
}

std::vector<float> OptModule::dataCost_occ(OptView view) {

	OptView source = view;
	OptView target = OtherView(source);
	int n_pixels = width_ * height_;
	int n_labels = 2;

	std::vector<float> cost;
	cost.resize(n_pixels * n_labels);
	//std::vector<bool> sanityCheck;	// sanity check
	//sanityCheck.resize(n_pixels, false);

	// data term - occ happens
	for (int pp_idx = 0; pp_idx < n_pixels; ++pp_idx) {
		cost[pp_idx * n_labels] = params_.lambda_occp;
	}

	// data term - visible
	for (int ss = 0; ss < segments_[source]->size(); ++ss) {
		std::vector<cv::Point2f> pixels = segments_[source]->at(ss).getPixelVector();
		cv::Mat state = segments_[source]->at(ss).getState();
		cv::Point2i tlPt = segments_[source]->at(ss).getCensusPatchTL();
		cv::Point2i brPt = segments_[source]->at(ss).getCensusPatchBR();
		std::vector<float> cost_noc = dataTerm_occ_cont(source, pixels, state, tlPt, brPt);
		for (int pp = 0; pp < pixels.size(); ++pp) {
			int pp_idx = (int)pixels[pp].x + width_ * (int)pixels[pp].y;
			cost[pp_idx * n_labels + 1] = cost_noc[pp];
			//sanityCheck[pp_idx] = true;	// sanity check
		}
	}

	//// sanity check
	//for (int pp_idx = 0; pp_idx < n_pixels; ++pp_idx) {
	//	if (sanityCheck[pp_idx] == false) {
	//		std::cout << " Error : OptModule.cpp = sanity check failure on Opt O" << std::endl;
	//	}
	//}

	// consistency term 
	std::vector<float> consistency_cost = consistencyTerm_occ(view);
	for (int pp_idx = 0; pp_idx < n_pixels; ++pp_idx) {
		cost[pp_idx * n_labels + 1] += consistency_cost[pp_idx];
	}

	// symmetry term
	std::vector<float> symmetry_cost = symmetryTerm_occ(view);
	for (int pp_idx = 0; pp_idx < n_pixels; ++pp_idx) {
		for (int ll = 0; ll < 2; ++ll) {
			int idx = pp_idx * n_labels + ll;
			cost[idx] += symmetry_cost[idx];
		}
	}

	return cost;
}

std::vector<float> OptModule::dataTerm_occ_cont(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state, const cv::Point2i& tlPoint, const cv::Point2i& brPoint) {

	OptView source = view;
	OptView target = OtherView(source);
	std::vector<cv::Point2f> tformedPixels;
	getTransformedPoints(pixels, state, tformedPixels);
	int sizePixels = (int)pixels.size();
	std::vector<float> cost;
	cost.resize(sizePixels, 0.f);

	// perspectiveCensus, create patch 
	int wPatch = brPoint.x - tlPoint.x + 2 * ternaryHalfWidth_ + 1;
	int hPatch = brPoint.y - tlPoint.y + 2 * ternaryHalfWidth_ + 1;
	std::vector<cv::Point2f> dstPatch, tformDstPatch;
	dstPatch.resize(wPatch*hPatch);
	tformDstPatch.resize(wPatch*hPatch);
	std::vector<bool> oobMask;
	oobMask.resize(wPatch*hPatch, true);
	int count_pp = 0;
	for (int hh = tlPoint.y - ternaryHalfWidth_; hh <= brPoint.y + ternaryHalfWidth_; ++hh) {
		for (int ww = tlPoint.x - ternaryHalfWidth_; ww <= brPoint.x + ternaryHalfWidth_; ++ww) {
			dstPatch[count_pp] = cv::Point2f((float)ww, (float)hh);
			count_pp++;
		}
	}

	// get transformed pts
	getTransformedPoints(dstPatch, state, tformDstPatch);

	// color interpolation
	std::vector<float> dstPatchVal;
	dstPatchVal.resize(tformDstPatch.size(), -1.f);
	for (int pp = 0; pp < tformDstPatch.size(); ++pp) {
		if (!isInside(target, tformDstPatch[pp].x, tformDstPatch[pp].y)) {
			oobMask[pp] = false;
			continue;
		}
		getInterpolatedPixel_single(gray_mat_[target], tformDstPatch[pp].x, tformDstPatch[pp].y, dstPatchVal[pp]);
	}

	for (int pp = 0; pp < sizePixels; ++pp) {

		if (isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {
			//cost[pp] = data_ternary_grad_rev1(source, target, oobMask, pixels[pp], tformedPixels[pp], tlPoint, wPatch, dstPatchVal, tformDstPatch);
			//cost[pp] = data_ternary_grad(source, target, pixels[pp], tformedPixels[pp], tlPoint, wPatch, dstPatchVal, tformDstPatch);
			//cost[pp] = data_intensity_grad(source, target, pixels[pp], tformedPixels[pp]);


			int idx_pp = (int)(pixels[pp].x + (int)pixels[pp].y * width_);

			// ternary
			std::vector<float> ternary1 = ternary_mat_cont[source][idx_pp];
			std::vector<float> ternary2;
			float hammDist_pre = 0.f;
			int n_oob = 0;

			{
				ternary2.resize(ternaryPatchSize_, ternaryOobCode_);

				int xCenter = pixels[pp].x - (tlPoint.x - ternaryHalfWidth_);
				int yCenter = pixels[pp].y - (tlPoint.y - ternaryHalfWidth_);

				float val_dst_c = dstPatchVal[xCenter + wPatch * yCenter];
				int idx_cc = 0;

				for (int offsetY = -ternaryHalfWidth_; offsetY <= ternaryHalfWidth_; ++offsetY){
					for (int offsetX = -ternaryHalfWidth_; offsetX <= ternaryHalfWidth_; ++offsetX) {

						if (ternary1[idx_cc] == ternaryOobCode_){
							++n_oob;
							++idx_cc;
							continue;
						}
						int pIdx = (xCenter + offsetX) + wPatch * (yCenter + offsetY);
						if (!oobMask[pIdx]) {
							++n_oob;
							++idx_cc;
							continue;
						}

						// hammingDist(ter1[idx_cc], ter2[idx_cc]) - Geman and McClure
						float val = ternary1[idx_cc] - ternary_sigmoid(dstPatchVal[pIdx] - val_dst_c);
						hammDist_pre += val * val / (0.1f + val * val);
						++idx_cc;
					}
				}
			}

			if (n_oob > (int)(ternaryPatchSize_ * 0.75f)){
				cost[pp] = params_.truncate_data;
				continue;
			}

			float perpixData = params_.alpha * std::min(hammDist_pre * (float)ternaryPatchSize_ / (float)(ternaryPatchSize_ - n_oob), params_.truncate_ternary);

			// gradient
			float gradtX = 0.f, gradtY = 0.f;
			getInterpolatedPixel_single_float(gradX_mat_[target], tformedPixels[pp].x, tformedPixels[pp].y, gradtX);
			getInterpolatedPixel_single_float(gradY_mat_[target], tformedPixels[pp].x, tformedPixels[pp].y, gradtY);
			float diffX = gradX_mat_[source].at<float>(pixels[pp]) - gradtX;
			float diffY = gradY_mat_[source].at<float>(pixels[pp]) - gradtY;
			float absDiff = std::min(fabsf(diffX) + fabsf(diffY), params_.truncate_grad);
			perpixData += (1.f - params_.alpha)*absDiff;

			cost[pp] = 5.08f * logf(1.f + perpixData*perpixData / 8.f);	// 5.08 is for setting offset, the lorentzian function pass (20,20)


		}
		else {
			cost[pp] = params_.truncate_data;
		}
	}

	return cost;
}

inline void OptModule::getInterpolatedPixel_single(const cv::Mat& image, float x, float y, float& val) const {

	int NW_x = std::min((int)x, width_ - 1);
	int NW_y = std::min((int)y, height_ - 1);
	int NE_x = std::min(NW_x + 1, width_ - 1);
	int NE_y = NW_y;
	int SW_x = NW_x;
	int SW_y = std::min(NW_y + 1, height_ - 1);
	int SE_x = std::min(NW_x + 1, width_ - 1);
	int SE_y = std::min(NW_y + 1, height_ - 1);

	float dx = x - NW_x;
	float dy = y - NW_y;

	if (NW_x >= 0 && NW_y >= 0 &&
		SE_x < width_ && SE_y < height_) {

		//uchar NW_v = image.at<uchar>(cv::Point(NW_x, NW_y));
		//uchar NE_v = image.at<uchar>(cv::Point(NE_x, NE_y));
		//uchar SW_v = image.at<uchar>(cv::Point(SW_x, SW_y));
		//uchar SE_v = image.at<uchar>(cv::Point(SE_x, SE_y));

		uchar NW_v = image.ptr<uchar>(NW_y)[NW_x];
		uchar NE_v = image.ptr<uchar>(NE_y)[NE_x];
		uchar SW_v = image.ptr<uchar>(SW_y)[SW_x];
		uchar SE_v = image.ptr<uchar>(SE_y)[SE_x];

		float NW_w = (1.f - dx)*(1.f - dy);
		float NE_w = dx*(1.f - dy);
		float SW_w = (1.f - dx)*dy;
		float SE_w = dx*dy;

		val = float(NW_v)*NW_w +
			float(NE_v)*NE_w +
			float(SW_v)*SW_w +
			float(SE_v)*SE_w;
	}
}

inline void OptModule::getInterpolatedPixel_single_float(const cv::Mat& image, float x, float y, float& val) const {

	int NW_x = std::min((int)x, width_ - 1);
	int NW_y = std::min((int)y, height_ - 1);
	int NE_x = std::min(NW_x + 1, width_ - 1);
	int NE_y = NW_y;
	int SW_x = NW_x;
	int SW_y = std::min(NW_y + 1, height_ - 1);
	int SE_x = std::min(NW_x + 1, width_ - 1);
	int SE_y = std::min(NW_y + 1, height_ - 1);

	float dx = x - NW_x;
	float dy = y - NW_y;

	if (NW_x >= 0 && NW_y >= 0 &&
		SE_x < width_ && SE_y < height_){

		//float NW_v = image.at<float>(cv::Point(NW_x, NW_y));
		//float NE_v = image.at<float>(cv::Point(NE_x, NE_y));
		//float SW_v = image.at<float>(cv::Point(SW_x, SW_y));
		//float SE_v = image.at<float>(cv::Point(SE_x, SE_y));

		float NW_v = image.ptr<float>(NW_y)[NW_x];
		float NE_v = image.ptr<float>(NE_y)[NE_x];
		float SW_v = image.ptr<float>(SW_y)[SW_x];
		float SE_v = image.ptr<float>(SE_y)[SE_x];

		float NW_w = (1.f - dx)*(1.f - dy);
		float NE_w = dx*(1.f - dy);
		float SW_w = (1.f - dx)*dy;
		float SE_w = dx*dy;

		val = NW_v*NW_w + NE_v*NE_w + SW_v*SW_w + SE_v*SE_w;
	}
}

float OptModule::pairwiseCost(OptView view, const int idx_spx1, const cv::Mat& state1, const int idx_spx2, const cv::Mat& state2, const int boundaryIndex) {

	std::vector<cv::Point2f> pixelsBoundary = boundaries_[view]->at(boundaryIndex).getPixelVector();
	std::vector<float> weightBoundaryPixels = boundaries_[view]->at(boundaryIndex).getPixelWeight();
	int sizeBoundary = (int)pixelsBoundary.size();

	float cost = 0.f;

	std::vector<cv::Point2f> H1_pts;
	std::vector<cv::Point2f> H2_pts;

	getTransformedPoints(pixelsBoundary, state1, H1_pts);
	getTransformedPoints(pixelsBoundary, state2, H2_pts);

	// calculating l2 term
	for (int pp = 0; pp < sizeBoundary; ++pp) {
		float weight = weightBoundaryPixels[pp];
		//float curr_energy = weight * std::min(abs(H1_pts[pp].x - H2_pts[pp].x) + abs(H1_pts[pp].y - H2_pts[pp].y), params_.truncate_pairwise);
		float curr_energy = weight * std::min(sqrtf((H1_pts[pp].x - H2_pts[pp].x) * (H1_pts[pp].x - H2_pts[pp].x) + (H1_pts[pp].y - H2_pts[pp].y) * (H1_pts[pp].y - H2_pts[pp].y)), params_.truncate_pairwise);
		cost += curr_energy;
	}

	return params_.lambda_pairwise * cost;
}

float OptModule::pairwiseCost_co_hi(OptView view, const int idx_spx1, const cv::Mat& state1, const int idx_spx2, const cv::Mat& state2, const int boundaryIndex) {

	std::vector<cv::Point2f> pixels1 = segments_[view]->at(idx_spx1).getPixelVector();
	std::vector<cv::Point2f> pixels2 = segments_[view]->at(idx_spx2).getPixelVector();
	std::vector<cv::Point2f> pixelsBoundary = boundaries_[view]->at(boundaryIndex).getPixelVector();
	std::vector<float> weightBoundaryPixels = boundaries_[view]->at(boundaryIndex).getPixelWeight();
	int sizePixel1 = (int)pixels1.size();
	int sizePixel2 = (int)pixels2.size();
	int sizeBoundary = (int)pixelsBoundary.size();

	float coPlanar_cost = 0.f;
	// calculating co-planar cost - l2 term
	{
		float energy1 = 0.f;
		std::vector<cv::Point2f> H1_pts1, H2_pts1;
		getTransformedPoints(pixels1, state1, H1_pts1);
		getTransformedPoints(pixels1, state2, H2_pts1);
		for (int pp = 0; pp < sizePixel1; ++pp) {
			energy1 += sqrtf((H1_pts1[pp].x - H2_pts1[pp].x) * (H1_pts1[pp].x - H2_pts1[pp].x) + (H1_pts1[pp].y - H2_pts1[pp].y) * (H1_pts1[pp].y - H2_pts1[pp].y));
		}

		float energy2 = 0.f;
		std::vector<cv::Point2f> H1_pts2, H2_pts2;
		getTransformedPoints(pixels2, state1, H1_pts2);
		getTransformedPoints(pixels2, state2, H2_pts2);
		for (int pp = 0; pp < sizePixel2; ++pp) {
			energy2 += sqrtf((H1_pts2[pp].x - H2_pts2[pp].x) * (H1_pts2[pp].x - H2_pts2[pp].x) + (H1_pts2[pp].y - H2_pts2[pp].y) * (H1_pts2[pp].y - H2_pts2[pp].y));
		}
		float bSpxWeight = boundaries_[view]->at(boundaryIndex).getSpxWeight();
		coPlanar_cost = bSpxWeight * (energy1 + energy2) / (float)(sizePixel1 + sizePixel2);
		//coPlanar_cost = (energy1 + energy2) / (float)(sizePixel1 + sizePixel2);
		// coPlanar_cost = bSpxWeight * energy1 / (float)sizePixel + energy2 / (float)sizePixel2;
	}

	float cost = 0.f;
	// calculating hinge cost - l2 term
	{
		std::vector<cv::Point2f> H1_pts;
		std::vector<cv::Point2f> H2_pts;
		getTransformedPoints(pixelsBoundary, state1, H1_pts);
		getTransformedPoints(pixelsBoundary, state2, H2_pts);
		for (int pp = 0; pp < sizeBoundary; ++pp) {
			float weight = weightBoundaryPixels[pp];
			//float curr_energy = weight * std::min(abs(H1_pts[pp].x - H2_pts[pp].x) + abs(H1_pts[pp].y - H2_pts[pp].y), params_.truncate_pairwise);
			float disp = sqrtf((H1_pts[pp].x - H2_pts[pp].x) * (H1_pts[pp].x - H2_pts[pp].x) + (H1_pts[pp].y - H2_pts[pp].y) * (H1_pts[pp].y - H2_pts[pp].y));
			float min_energy = std::min(params_.truncate_pairwise, std::min(params_.weight_coPlanar * coPlanar_cost, params_.truncate_coPlanar + disp));
			float curr_energy = weight * min_energy;
			cost += curr_energy;
		}
	}

	return params_.lambda_pairwise * cost;
}

float OptModule::consistencyTerm(OptView view, const std::vector<cv::Point2f>& pixels, const cv::Mat& state) {

	float cost = 0.f;
	OptView source = view;
	OptView target = OtherView(source);
	std::vector<cv::Point2f> tformedPixels;
	getTransformedPoints(pixels, state, tformedPixels);
	int sizePixels = (int)pixels.size();

	for (int pp = 0; pp < sizePixels; ++pp) {

		if (isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {

			// [occVal] true(255) = visible, false(0) = occluded
			bool occVal_src = (bool)(occMask_[source].at<uchar>((int)pixels[pp].y, (int)pixels[pp].x));
			bool occVal_tgt = (bool)(occMask_[target].at<uchar>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f)));

			// not occluded
			if (occVal_src == true && occVal_tgt == true) {

				int spx_idx_tgt = (int)superpixelLabelMap_[target].at<unsigned short>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f));
				cv::Mat state_tgt = segments_[target]->at(spx_idx_tgt).getState();
				cv::Point2f pp_backprojected;
				getTransformedPoint(tformedPixels[pp], state_tgt, pp_backprojected);	// maybe revise ? not using function

				float dist = sqrtf((pixels[pp].x - pp_backprojected.x)*(pixels[pp].x - pp_backprojected.x) + (pixels[pp].y - pp_backprojected.y)*(pixels[pp].y - pp_backprojected.y));

				cost += std::min(dist, params_.truncate_consistency);
			}
		}
	}

	return params_.lambda_consistency * cost;
}

std::vector<float> OptModule::consistencyTerm_occ(OptView view) {

	OptView source = view;
	OptView target = OtherView(source);
	std::vector<float> cost;
	cost.resize(height_ * width_, 0.f);

	// consistency - forward
	for (int ss = 0; ss < segments_[source]->size(); ++ss) {

		std::vector<cv::Point2f> pixels = segments_[source]->at(ss).getPixelVector();
		cv::Mat state = segments_[source]->at(ss).getState();

		std::vector<cv::Point2f> tformedPixels;
		getTransformedPoints(pixels, state, tformedPixels);
		int sizePixels = (int)pixels.size();

		for (int pp = 0; pp < sizePixels; ++pp) {

			if (isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {

				// [occVal] true(255) = visible, false(0) = occluded
				bool occVal_tgt = (bool)(occMask_[target].at<uchar>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f)));

				// not occluded
				if (occVal_tgt == true) {	// occVal_src == true (assumption)

					int spx_idx_tgt = (int)superpixelLabelMap_[target].at<unsigned short>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f));
					cv::Mat state_tgt = segments_[target]->at(spx_idx_tgt).getState();
					cv::Point2f pp_backprojected;
					getTransformedPoint(tformedPixels[pp], state_tgt, pp_backprojected);

					float dist = sqrtf((pixels[pp].x - pp_backprojected.x)*(pixels[pp].x - pp_backprojected.x) + (pixels[pp].y - pp_backprojected.y)*(pixels[pp].y - pp_backprojected.y));
					int pp_idx = pixels[pp].x + width_ * (int)pixels[pp].y;
					cost[pp_idx] = params_.lambda_consistency * std::min(dist, params_.truncate_consistency);
				}
			}
		}
	}

	// consistency - backward
	for (int ss = 0; ss < segments_[target]->size(); ++ss) {

		std::vector<cv::Point2f> pixels = segments_[target]->at(ss).getPixelVector();
		cv::Mat state = segments_[target]->at(ss).getState();

		std::vector<cv::Point2f> tformedPixels;
		getTransformedPoints(pixels, state, tformedPixels);
		int sizePixels = (int)pixels.size();

		for (int pp = 0; pp < sizePixels; ++pp) {

			if (isInside(source, tformedPixels[pp].x, tformedPixels[pp].y)) {

				// [occVal] true(255) = visible, false(0) = occluded
				bool occVal_tgt = (bool)(occMask_[target].at<uchar>((int)pixels[pp].y, (int)pixels[pp].x));

				// not occluded
				if (occVal_tgt == true) {	// occVal_src == true (assumption)

					int spx_idx_src = (int)superpixelLabelMap_[source].at<unsigned short>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f));
					cv::Mat state_src = segments_[source]->at(spx_idx_src).getState();
					cv::Point2f pp_backprojected;
					getTransformedPoint(tformedPixels[pp], state_src, pp_backprojected);

					float dist = sqrtf((pixels[pp].x - pp_backprojected.x)*(pixels[pp].x - pp_backprojected.x) + (pixels[pp].y - pp_backprojected.y)*(pixels[pp].y - pp_backprojected.y));
					int pp_idx = (int)(tformedPixels[pp].x + 0.5f) + width_ * (int)(tformedPixels[pp].y + 0.5f);
					cost[pp_idx] += params_.lambda_consistency * std::min(dist, params_.truncate_consistency);
				}
			}
		}
	}

	return cost;
}

void OptModule::updateConsistencyEnergyMap(OptView view, const std::set<int>& set_in) {

	OptView source = view;
	OptView target = OtherView(source);
	float max_val = params_.truncate_consistency * 2.0f;
	// consistency - forward
	for (auto iter = set_in.begin(); iter != set_in.end(); ++iter) {

		std::vector<cv::Point2f> pixels = segments_[source]->at(*iter).getPixelVector();
		cv::Mat state = segments_[source]->at(*iter).getState();
		std::vector<cv::Point2f> tformedPixels;
		getTransformedPoints(pixels, state, tformedPixels);
		int sizePixels = (int)pixels.size();

		for (int pp = 0; pp < sizePixels; ++pp) {

			if (!isInside(target, tformedPixels[pp].x, tformedPixels[pp].y)) {
				consistencyEnergy_[source].at<float>(pixels[pp]) = max_val;
				continue;
			}

			// [occVal] true(255) = visible, false(0) = occluded

			bool occVal_src = (bool)(occMask_[source].at<uchar>((int)pixels[pp].y, (int)pixels[pp].x));
			if (occVal_src == false){
				consistencyEnergy_[source].at<float>(pixels[pp]) = max_val;
				continue;
			}

			bool occVal_tgt = (bool)(occMask_[target].at<uchar>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f)));
			if (occVal_tgt == false) {
				consistencyEnergy_[source].at<float>(pixels[pp]) = max_val;
				continue;
			}

			// not occluded
			int spx_idx_tgt = (int)superpixelLabelMap_[target].at<unsigned short>((int)(tformedPixels[pp].y + 0.5f), (int)(tformedPixels[pp].x + 0.5f));
			cv::Mat state_tgt = segments_[target]->at(spx_idx_tgt).getState();
			cv::Point2f pp_backprojected;
			getTransformedPoint(tformedPixels[pp], state_tgt, pp_backprojected);

			float dist = sqrtf((pixels[pp].x - pp_backprojected.x)*(pixels[pp].x - pp_backprojected.x) + (pixels[pp].y - pp_backprojected.y)*(pixels[pp].y - pp_backprojected.y));
			consistencyEnergy_[source].at<float>(pixels[pp]) = std::min(dist, params_.truncate_consistency);
		}
	}

	return;
}

std::vector<float> OptModule::symmetryTerm_occ(OptView view) {

	OptView source = view;
	OptView target = OtherView(source);
	std::vector<float> cost;
	int n_pixels = height_ * width_;
	int n_labels = 2;
	cost.resize(n_pixels * n_labels, 0.f);

	for (int hh = 0; hh < height_; ++hh){
		for (int ww = 0; ww < width_; ++ww) {

			bool occupied = (bool)projectionMask_[source].at<uchar>(hh, ww);

			int pp_idx_0 = (ww + width_ * hh) * n_labels;
			int pp_idx_1 = (ww + width_ * hh) * n_labels + 1;

			// if something is occupying, "should be visible" => occMask = 255. if not, then cost occurs
			if (occupied) {
				cost[pp_idx_0] = params_.lambda_symmetry;
			}
			else {	// not occupied -> disocc occurs. Thus it should be occluded. -> if not occluded, then cost occurs.
				cost[pp_idx_1] = params_.lambda_symmetry;
			}
		}
	}
	return cost;
}

void OptModule::projectionIntoTheOtherView(OptView view) {

	OptView source = view;
	OptView target = OtherView(source);

	projectionMask_[target].setTo(0);

	// Forawrd Warping 
	int sizeSegment = (int)segments_[source]->size();

	for (int ss = 0; ss < sizeSegment; ++ss) {

		std::vector<cv::Point2f> pixels = segments_[source]->at(ss).getPixelVector();
		cv::Mat currState = segments_[source]->at(ss).getState();

		// corner points
		cv::Point2i pTL = segments_[source]->at(ss).getCensusPatchTL();
		cv::Point2i pBR = segments_[source]->at(ss).getCensusPatchBR();

		std::vector<cv::Point2f> pCorners, pCorners_tf;
		pCorners.resize(4);
		pCorners[0] = cv::Point2f((float)pTL.x - 1.f, (float)pTL.y - 1.f);
		pCorners[1] = cv::Point2f((float)pBR.x + 1.f, (float)pBR.y + 1.f);
		pCorners[2] = cv::Point2f(pCorners[1].x, pCorners[0].y);		// pTR
		pCorners[3] = cv::Point2f(pCorners[0].x, pCorners[1].y);		// pBL
		getTransformedPoints(pCorners, currState, pCorners_tf);

		float max_x = 0.f;
		float max_y = 0.f;
		float min_x = FLT_MAX;
		float min_y = FLT_MAX;

		for (int nn = 0; nn < 4; ++nn) {
			max_x = std::max(max_x, pCorners_tf[nn].x);
			max_y = std::max(max_y, pCorners_tf[nn].y);
			min_x = std::min(min_x, pCorners_tf[nn].x);
			min_y = std::min(min_y, pCorners_tf[nn].y);
		}

		int min_xx = min_x - 1;
		int min_yy = min_y - 1;
		int max_xx = max_x + 1;
		int max_yy = max_y + 1;

		//// validity condition
		//if (min_xx >= max_xx || min_yy >= max_yy){
		//	continue;
		//}
		//if ((max_xx - min_xx) > 1500 || (max_yy - min_yy) > 1500){
		//	continue;
		//}

		// creating patch in the dst image & its inverse warping to src image
		std::vector<cv::Point2f> patch_dst, H1_inv_patch_dst;
		patch_dst.resize((max_xx - min_xx + 1)*(max_yy - min_yy + 1));
		int count_dst = 0;
		for (int xx = min_xx; xx <= max_xx; ++xx){
			for (int yy = min_yy; yy <= max_yy; ++yy) {
				patch_dst[count_dst] = cv::Point2f((float)xx, (float)yy);
				count_dst++;
			}
		}
		getTransformedPoints(patch_dst, currState.inv(), H1_inv_patch_dst);

		for (int pp = 0; pp < H1_inv_patch_dst.size(); ++pp) {

			// inside check in the both view
			if (!isInside(target, patch_dst[pp].x, patch_dst[pp].y)) {
				continue;
			}
			if (!isInside(source, H1_inv_patch_dst[pp].x, H1_inv_patch_dst[pp].y)) {
				continue;
			}

			// label check in the src image
			int x_src = (int)(H1_inv_patch_dst[pp].x + 0.5f);
			int y_src = (int)(H1_inv_patch_dst[pp].y + 0.5f);
			int idxSp = (int)(superpixelLabelMap_[view].at<unsigned short>(y_src, x_src));
			if (idxSp != ss) {
				continue;
			}

			projectionMask_[target].at<uchar>((int)(patch_dst[pp].y + 0.5f), (int)(patch_dst[pp].x + 0.5f)) = (uchar)255;
		}
	}

	// hole filling
	cv::Mat tempMask = projectionMask_[target].clone();
	for (int hh = 0; hh < height_; ++hh){
		for (int ww = 0; ww < width_; ++ww){

			if (tempMask.at<uchar>(hh, ww) == 0){
				bool flag = false;
				if ((hh != 0) && (tempMask.at<uchar>(hh - 1, ww) != 0)){
					flag = true;
				}
				if ((hh != height_ - 1) && (tempMask.at<uchar>(hh + 1, ww) != 0)){
					flag = true;
				}
				if ((ww != 0) && (tempMask.at<uchar>(hh, ww - 1) != 0)){
					flag = true;
				}
				if ((ww != width_ - 1) && (tempMask.at<uchar>(hh, ww + 1) != 0)){
					flag = true;
				}

				if (flag){
					projectionMask_[target].at<uchar>(hh, ww) = (uchar)255;
				}
			}
		}
	}

}

void OptModule::getSignedGradientImage(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY) {

	cv::Mat srcImage, srcGray;
	srcImage = image.clone();
	gradX = cv::Mat(srcImage.rows, srcImage.cols, CV_32F);
	gradY = cv::Mat(srcImage.rows, srcImage.cols, CV_32F);
	cv::cvtColor(srcImage, srcGray, CV_BGR2GRAY);

	for (int i = 0; i < srcImage.cols; ++i) {
		for (int j = 0; j < srcImage.rows; ++j) {

			uchar grey_center = srcGray.at<uchar>(cv::Point(i, j));

			uchar grey_left = grey_center;
			uchar grey_right = grey_center;
			uchar grey_top = grey_center;
			uchar grey_bottom = grey_center;

			if (i != 0) {
				grey_left = srcGray.at<uchar>(cv::Point(i - 1, j));
			}

			if (i != srcImage.cols - 1) {
				grey_right = srcGray.at<uchar>(cv::Point(i + 1, j));
			}

			if (j != 0) {
				grey_top = srcGray.at<uchar>(cv::Point(i, j - 1));
			}

			if (j != srcImage.rows - 1) {
				grey_bottom = srcGray.at<uchar>(cv::Point(i, j + 1));
			}

			float delta_x = (float)grey_right - (float)grey_left;
			float delta_y = (float)grey_bottom - (float)grey_top;

			gradX.at<float>(cv::Point(i, j)) = delta_x / 2.f;
			gradY.at<float>(cv::Point(i, j)) = delta_y / 2.f;
		}
	}
	return;
}

cv::Mat OptModule::getCensusImage(const cv::Mat& image) {

	int censusWindowRadius = 3;
	float offset = 0.3f;

	cv::Mat srcImage, srcGray, dstImage;
	srcImage = image.clone();
	dstImage = cv::Mat(srcImage.rows, srcImage.cols, CV_64F);
	cv::cvtColor(srcImage, srcGray, CV_BGR2GRAY);

	for (int hh = 0; hh < image.rows; ++hh) {
		for (int ww = 0; ww < image.cols; ++ww) {

			float centerValue = (float)srcGray.at<uchar>(hh, ww);

			double censusCode = 0.0;
			for (int offsetY = -censusWindowRadius; offsetY <= censusWindowRadius; ++offsetY) {
				for (int offsetX = -censusWindowRadius; offsetX <= censusWindowRadius; ++offsetX) {
					censusCode = censusCode * 2;
					if (hh + offsetY >= 0 && hh + offsetY < image.rows
						&& ww + offsetX >= 0 && ww + offsetX < image.cols) {

						int val = 0;
						if ((float)(srcGray.at<uchar>(hh + offsetY, ww + offsetX)) >= centerValue + offset) {
							val = 1;
						}
						censusCode += val;
					}
				}
			}
			dstImage.at<double>(hh, ww) = censusCode;
		}
	}
	return dstImage;
}

cv::Mat OptModule::getTernaryImage(const cv::Mat& image) {

	int censusWindowRadius = 3;
	float offset = 1.25f;

	cv::Mat srcImage, srcGray, dstImage;
	srcImage = image.clone();
	dstImage = cv::Mat(srcImage.rows, srcImage.cols, CV_64F);
	cv::cvtColor(srcImage, srcGray, CV_BGR2GRAY);

	for (int hh = 0; hh < image.rows; ++hh) {
		for (int ww = 0; ww < image.cols; ++ww) {

			float centerValue = (float)srcGray.at<uchar>(hh, ww);

			double censusCode = 0.0;
			for (int offsetY = -censusWindowRadius; offsetY <= censusWindowRadius; ++offsetY) {
				for (int offsetX = -censusWindowRadius; offsetX <= censusWindowRadius; ++offsetX) {
					censusCode = censusCode * 3.0;
					if (hh + offsetY >= 0 && hh + offsetY < image.rows
						&& ww + offsetX >= 0 && ww + offsetX < image.cols) {

						double val = 0.0;
						if ((float)(srcGray.at<uchar>(hh + offsetY, ww + offsetX)) >= centerValue + offset) {
							val = 2.0;
						}
						else if ((float)(srcGray.at<uchar>(hh + offsetY, ww + offsetX)) <= centerValue - offset) {
							val = 0.0;
						}
						else {
							val = 1.0;
						}
						censusCode += val;
					}
				}
			}
			dstImage.at<double>(hh, ww) = censusCode;
		}
	}
	return dstImage;
}

void OptModule::getTernaryImage_continuous(const cv::Mat& image, std::vector<std::vector<float> >& ternary_mat_cont_in) {

	cv::Mat srcImage, srcGray;
	srcImage = image.clone();
	cv::cvtColor(srcImage, srcGray, CV_BGR2GRAY);
	ternary_mat_cont_in.resize(srcImage.rows* srcImage.cols);
	for (int ii = 0; ii < srcImage.rows* srcImage.cols; ++ii){
		ternary_mat_cont_in[ii].resize(ternaryPatchSize_);
	}

	for (int hh = 0; hh < srcImage.rows; ++hh) {
		for (int ww = 0; ww < srcImage.cols; ++ww) {

			float centerValue = (float)srcGray.at<uchar>(hh, ww);
			int idx_pp = ww + hh * srcImage.cols;
			int idx_cc = 0;

			for (int offsetY = -ternaryHalfWidth_; offsetY <= ternaryHalfWidth_; ++offsetY) {
				for (int offsetX = -ternaryHalfWidth_; offsetX <= ternaryHalfWidth_; ++offsetX) {

					if (hh + offsetY >= 0 && hh + offsetY < srcImage.rows
						&& ww + offsetX >= 0 && ww + offsetX < srcImage.cols) {
						float val = (float)(srcGray.at<uchar>(hh + offsetY, ww + offsetX)) - centerValue;
						ternary_mat_cont_in[idx_pp][idx_cc] = ternary_sigmoid(val);
					}
					else{
						ternary_mat_cont_in[idx_pp][idx_cc] = ternaryOobCode_;
					}
					idx_cc++;
				}
			}
		}
	}
	return;
}

void OptModule::getTransformedPoints(const std::vector<cv::Point2f>& srcPoints, const cv::Mat& state, std::vector<cv::Point2f>& dstPoints) {

	// faster version -- 5.5(s) -> 1.7(s)
	int size = (int)srcPoints.size();
	if (size == 0) {
		std::cout << " ERROR imageOperator.cpp : srcPoints size is 0" << std::endl;
	}
	dstPoints.resize(size);

	for (int pp = 0; pp < size; ++pp) {
		float normconst = state.ptr<float>(2)[0] * srcPoints[pp].x + state.ptr<float>(2)[1] * srcPoints[pp].y + state.ptr<float>(2)[2];
		dstPoints[pp].x = (state.ptr<float>(0)[0] * srcPoints[pp].x + state.ptr<float>(0)[1] * srcPoints[pp].y + state.ptr<float>(0)[2]) / normconst;
		dstPoints[pp].y = (state.ptr<float>(1)[0] * srcPoints[pp].x + state.ptr<float>(1)[1] * srcPoints[pp].y + state.ptr<float>(1)[2]) / normconst;
	}
}

void OptModule::getTransformedPoint(const cv::Point2f& srcPoints, const cv::Mat& state, cv::Point2f& dstPoints) {

	// faster version -- 5.5(s) -> 1.7(s)
	float normconst = state.ptr<float>(2)[0] * srcPoints.x + state.ptr<float>(2)[1] * srcPoints.y + state.ptr<float>(2)[2];
	dstPoints.x = (state.ptr<float>(0)[0] * srcPoints.x + state.ptr<float>(0)[1] * srcPoints.y + state.ptr<float>(0)[2]) / normconst;
	dstPoints.y = (state.ptr<float>(1)[0] * srcPoints.x + state.ptr<float>(1)[1] * srcPoints.y + state.ptr<float>(1)[2]) / normconst;
}

bool OptModule::findCorrespondencebyWarping(OptView view, int index, std::vector<cv::Point2f>& srcP, std::vector<cv::Point2f>& dstP) {

	OptView source = view;				// time = t
	OptView target = OtherView(source);	// time = t+1


	cv::Point2i topLeftP = segments_[source]->at(index).getBboxTLpoint();
	cv::Size bbox = segments_[source]->at(index).getBboxSize();

	cv::Point2f bBoxCenterP = cv::Point2f(topLeftP.x + bbox.width / 2.f, topLeftP.y + bbox.height / 2.f);
	cv::Point2f dstbBoxCenterP;

	// bbox setting
	cv::Mat currState = segments_[source]->at(index).getState();
	getTransformedPoint(bBoxCenterP, currState, dstbBoxCenterP);

	cv::Point2i dstTopLeftP = cv::Point2i((int)(dstbBoxCenterP.x - bbox.width / 2.f), (int)(dstbBoxCenterP.y - bbox.height / 2.f));
	cv::Point2i dstBottomRightP = cv::Point2i(dstTopLeftP.x + bbox.width, dstTopLeftP.y + bbox.height);

	if (dstTopLeftP.x > width_ - 1 || dstBottomRightP.x < 0 || dstTopLeftP.y > height_ - 1 || dstBottomRightP.y < 0){
		return false;
	}

	dstTopLeftP.x = std::max(0, dstTopLeftP.x);
	dstTopLeftP.y = std::max(0, dstTopLeftP.y);

	if (dstBottomRightP.x > width_ - 1){
		dstTopLeftP.x = width_ - 1 - bbox.width;
	}
	if (dstBottomRightP.y > height_ - 1){
		dstTopLeftP.y = height_ - 1 - bbox.height;
	}

	// lucas-kanade
	std::vector<cv::Point2f> srcFeatures = segments_[source]->at(index).getFeatureToTrack();
	std::vector<cv::Point2f> dstFeatures;
	std::vector<uchar> featuresFound;
	cv::Mat err;

	// patch
	cv::Mat srcPatch = segments_[source]->at(index).getImagePatch();
	cv::Mat dstPatch = gray_mat_[target](cv::Rect(dstTopLeftP.x, dstTopLeftP.y, bbox.width, bbox.height)).clone();

	if ((int)srcFeatures.size() < 4){
		return false;
	}

	// sanity check
	if (srcPatch.size() != dstPatch.size()){
		std::cout << " Fatal Error : different patch size" << std::endl;
	}

	// revise it later - parameter setting
	cv::calcOpticalFlowPyrLK(srcPatch, dstPatch, srcFeatures, dstFeatures, featuresFound, err, cv::Size(bbox.width*0.8, bbox.width*0.8), 1);

	// displaying
	//cv::Mat srcPatch_copy = srcPatch.clone();
	//cv::Mat dstPatch_copy = dstPatch.clone();

	//Draw lines connecting previous position and current position
	srcP.resize(0);
	dstP.resize(0);
	for (int pp = 0; pp < (int)dstFeatures.size(); pp++){
		if (featuresFound[pp]){
			srcP.push_back(cv::Point2f(srcFeatures[pp].x + topLeftP.x, srcFeatures[pp].y + topLeftP.y));
			dstP.push_back(cv::Point2f(dstFeatures[pp].x + dstTopLeftP.x, dstFeatures[pp].y + dstTopLeftP.y));
			// test
			//cv::line(srcPatch, srcFeatures[pp], dstFeatures[pp], cv::Scalar(0, 0, 255), 4);
			//cv::line(dstPatch, dstFeatures[pp], dstFeatures[pp], cv::Scalar(0, 255, 0), 4);
		}
	}

	//cv::imshow("srcPatch", srcPatch_copy);
	//cv::imshow("dstPatch", dstPatch_copy);
	//cv::waitKey(0);
	//cv::destroyWindow("srcPatch");
	//cv::destroyWindow("dstPatch");

	//std::cout << (int)srcP.size() << std::endl;
	if ((int)srcP.size() < 4){
		return false;
	}

	return true;
}

bool OptModule::isInside(OptView view, float x, float y) const {

	if (!std::isfinite(x) || !std::isfinite(y)) {
		return false;
	}

	if (x < 0 || x > width_ - 1 || y < 0 || y > height_ - 1) {
		return false;
	}
	else {
		return true;
	}
}

bool OptModule::isStateValid(OptView view, int index, const cv::Mat& state){

	if (state.empty()){
		return false;
	}

	// variable definition
	float max_motion_sqr = 250.f *250.f;
	std::vector<cv::Point2f> vecCornerPts = segments_[view]->at(index).getVecCornerPts();
	std::vector<cv::Point2f> vecCornerPts_tf;
	getTransformedPoints(vecCornerPts, state, vecCornerPts_tf);

	std::vector<float> vec_dxx, vec_dyy;

	// NaN
	//for (int ii = 0; ii < vecCornerPts_tf.size(); ++ii){
	//	if (std::isnan(vecCornerPts_tf[ii].x) || std::isnan(vecCornerPts_tf[ii].y)){
	//		return false;
	//	}
	//}
	if (std::isnan(vecCornerPts_tf[0].x) || std::isnan(vecCornerPts_tf[0].y)){
		return false;
	}

	for (int ii = 0; ii < vecCornerPts.size(); ++ii){

		float dxx = fabsf(vecCornerPts_tf[ii].x - vecCornerPts[ii].x);
		float dyy = fabsf(vecCornerPts_tf[ii].y - vecCornerPts[ii].y);
		vec_dxx.push_back(dxx);
		vec_dyy.push_back(dyy);	// maybe revise?

		// Check that the displacement is not more than the max motion
		if (dxx*dxx + dyy*dyy > max_motion_sqr){
			return false;
		}
	}

	float max_expansion = 50.f;

	// Check whether the expansion size is exceeded
	if (fabsf(vec_dxx[0] - vec_dxx[1]) > max_expansion){	// x-displacement between TL and TR
		return false;
	}
	if (fabsf(vec_dxx[2] - vec_dxx[3]) > max_expansion){	// x-displacement between BL and BR
		return false;
	}
	if (fabsf(vec_dyy[0] - vec_dyy[2]) > max_expansion){	// y-displacement between TL and BL
		return false;
	}
	if (fabsf(vec_dyy[1] - vec_dyy[3]) > max_expansion){	// y-displacement between TR and BR
		return false;
	}

	//float dxTop = abs(vecCornerPts_tf[0].x - vecCornerPts_tf[1].x);
	//float dxBottom = abs(vecCornerPts_tf[2].x - vecCornerPts_tf[3].x);
	//float dyLeft = abs(vecCornerPts_tf[0].y - vecCornerPts_tf[2].y);
	//float dyRight = abs(vecCornerPts_tf[1].y - vecCornerPts_tf[3].y);

	float squeezeThreshold = 1.0f;

	// Check whether it is sqeezed
	if (fabsf(vecCornerPts_tf[0].x - vecCornerPts_tf[1].x) < squeezeThreshold){	// x-displacement between TL and TR
		return false;
	}
	if (fabsf(vecCornerPts_tf[2].x - vecCornerPts_tf[3].x) < squeezeThreshold){	// x-displacement between BL and BR
		return false;
	}
	if (fabsf(vecCornerPts_tf[0].y - vecCornerPts_tf[2].y) < squeezeThreshold){	// y-displacement between TL and BL
		return false;
	}
	if (fabsf(vecCornerPts_tf[1].y - vecCornerPts_tf[3].y) < squeezeThreshold){	// y-displacement between TR and BR
		return false;
	}

	return true;
}

void OptModule::readParams(const std::string& fileName) {

	cv::FileStorage fs(fileName, cv::FileStorage::READ);

	if (!fs.isOpened()) {
		std::cout << " WARNING optimizationModule.cpp : Cannot find the parameter file = " << fileName << std::endl;
		log_file_ << " WARNING optimizationModule.cpp : Cannot find the parameter file = " << fileName << std::endl;
		return;
	}

	if (!fs["verbose"].empty()) {
		fs["verbose"] >> params_.verbose;
	}

	if (!fs["CNN_feature_Extract"].empty()) {
		fs["CNN_feature_Extract"] >> params_.CNN_feature_Extract;
	}

	// region definition
	if (!fs["region_width"].empty()) {
		fs["region_width"] >> params_.region_width;
	}
	if (!fs["region_height"].empty()) {
		fs["region_height"] >> params_.region_height;
	}
	if (!fs["fix_boundary_region"].empty()) {
		fs["fix_boundary_region"] >> params_.fix_boundary_region;
	}
	if (!fs["n_intervals"].empty()) {
		fs["n_intervals"] >> params_.n_intervals;
	}
	if (!fs["superpixelNum"].empty()) {
		fs["superpixelNum"] >> params_.superpixelNum;
	}
	if (!fs["region_overlap_ratio"].empty()) {
		fs["region_overlap_ratio"] >> params_.region_overlap_ratio;
	}

	// opt
	if (!fs["n_outer_iterations"].empty()) {
		fs["n_outer_iterations"] >> params_.n_outer_iterations;
	}
	if (!fs["n_random_states"].empty()) {
		fs["n_random_states"] >> params_.n_random_states;
	}
	if (!fs["n_random_states_all"].empty()) {
		fs["n_random_states_all"] >> params_.n_random_states_all;
	}
	if (!fs["size_perturb_scale"].empty()) {
		fs["size_perturb_scale"] >> params_.size_perturb_scale;
	}
	if (!fs["size_perturb_trans"].empty()) {
		fs["size_perturb_trans"] >> params_.size_perturb_trans;
	}

	if (!fs["bidirectional"].empty()) {
		fs["bidirectional"] >> params_.bidirectional;
	}

	if (!fs["alpha"].empty()) {
		fs["alpha"] >> params_.alpha;
	}

	if (!fs["lambda_pairwise"].empty()) {
		fs["lambda_pairwise"] >> params_.lambda_pairwise;
	}
	if (!fs["lambda_occPott"].empty()) {
		fs["lambda_occPott"] >> params_.lambda_occPott;
	}
	if (!fs["lambda_consistency"].empty()) {
		fs["lambda_consistency"] >> params_.lambda_consistency;
	}
	if (!fs["lambda_symmetry"].empty()) {
		fs["lambda_symmetry"] >> params_.lambda_symmetry;
	}

	if (!fs["lambda_occp"].empty()) {
		fs["lambda_occp"] >> params_.lambda_occp;
	}

	if (!fs["truncate_ternary"].empty()) {
		fs["truncate_ternary"] >> params_.truncate_ternary;
	}
	if (!fs["truncate_grad"].empty()) {
		fs["truncate_grad"] >> params_.truncate_grad;
	}
	params_.truncate_data = params_.alpha * params_.truncate_ternary + (1 - params_.alpha) * params_.truncate_grad;

	if (!fs["weight_coPlanar"].empty()) {
		fs["weight_coPlanar"] >> params_.weight_coPlanar;
	}
	if (!fs["truncate_coPlanar"].empty()) {
		fs["truncate_coPlanar"] >> params_.truncate_coPlanar;
	}
	if (!fs["truncate_pairwise"].empty()) {
		fs["truncate_pairwise"] >> params_.truncate_pairwise;
	}
	if (!fs["truncate_consistency"].empty()) {
		fs["truncate_consistency"] >> params_.truncate_consistency;
	}
	if (!fs["bPixelWeight"].empty()) {
		fs["bPixelWeight"] >> params_.bPixelWeight;
	}
	if (!fs["superpixelWeight"].empty()) {
		fs["superpixelWeight"] >> params_.superpixelWeight;
	}

	if (!fs["truncate_CNNF"].empty()) {
		fs["truncate_CNNF"] >> params_.truncate_CNNF;
	}

	fs.release();
}

void OptModule::PrintParameter() {

	std::cout << "verbose " << params_.verbose << std::endl;
	std::cout << "n_outer_iterations " << params_.n_outer_iterations << std::endl;
	std::cout << "n_random_states " << params_.n_random_states << std::endl;
	std::cout << "size_perturb_scale " << params_.size_perturb_scale << std::endl;
	std::cout << "size_perturb_trans " << params_.size_perturb_trans << std::endl;
	std::cout << "bidirectional " << params_.bidirectional << std::endl;
	std::cout << "alpha " << params_.alpha << std::endl;
	std::cout << "lambda_pairwise " << params_.lambda_pairwise << std::endl;
	std::cout << "lambda_occPott " << params_.lambda_occPott << std::endl;
	std::cout << "lambda_consistency " << params_.lambda_consistency << std::endl;
	std::cout << "lambda_symmetry " << params_.lambda_symmetry << std::endl;
	std::cout << "lambda_occp " << params_.lambda_occp << std::endl;
	std::cout << "truncate_ternary " << params_.truncate_ternary << std::endl;
	std::cout << "truncate_grad " << params_.truncate_grad << std::endl;
	std::cout << "truncate_pairwise " << params_.truncate_pairwise << std::endl;
	std::cout << "truncate_consistency " << params_.truncate_consistency << std::endl;
	std::cout << "bPixelWeight " << params_.bPixelWeight << std::endl;
	std::cout << "superpixelWeight " << params_.superpixelWeight << std::endl;

}

float OptModule::getEnergyDataTerm(OptView view){

	float energy = 0.f;
	int n_segments = (int)(segments_[view]->size());

	for (int ss = 0; ss < n_segments; ++ss) {

		std::vector<cv::Point2f> pixels = segments_[view]->at(ss).getPixelVector();
		cv::Point2i tlPt = segments_[view]->at(ss).getCensusPatchTL();
		cv::Point2i brPt = segments_[view]->at(ss).getCensusPatchBR();
		cv::Mat state_ss = segments_[view]->at(ss).getState();

		energy += dataTerm_cont(view, pixels, state_ss, tlPt, brPt);
	}
	return energy;
}

float OptModule::getEnergyPairwiseTerm(OptView view){

	float energy = 0.f;
	int n_boundary = boundaries_[view]->size();

	for (int bb = 0; bb < n_boundary; ++bb) {

		int seg1 = boundaries_[view]->at(bb).getSegmentIndices().first;
		int seg2 = boundaries_[view]->at(bb).getSegmentIndices().second;

		// calculate the pairwise cost between seg1 and seg2 
		cv::Mat state1 = segments_[view]->at(seg1).getState();
		cv::Mat state2 = segments_[view]->at(seg2).getState();

		//energy += pairwiseCost(view, seg1, state1, seg2, state2, bb);
		energy += pairwiseCost_co_hi(view, seg1, state1, seg2, state2, bb);
	}
	return energy;
}

float OptModule::getEnergyConsistencyTerm(OptView view){

	float energy = 0.f;
	int n_segments = (int)(segments_[view]->size());

	for (int ss = 0; ss < n_segments; ++ss) {

		std::vector<cv::Point2f> pixels = segments_[view]->at(ss).getPixelVector();
		cv::Mat state_ss = segments_[view]->at(ss).getState();

		energy += consistencyTerm(view, pixels, state_ss);
	}

	return energy;
}

float OptModule::getEnergySymmetryTerm(OptView view){

	int count = 0;
	for (int hh = 0; hh < height_; ++hh){
		for (int ww = 0; ww < width_; ++ww){
			bool occupied = (bool)projectionMask_[view].at<uchar>(hh, ww);
			bool occVal = (bool)(occMask_[view].at<uchar>(hh, ww));

			if (occupied != occVal){
				count++;
			}
		}
	}
	return params_.lambda_symmetry * count;
}

float OptModule::getEnergyOccPairwiseTerm(OptView view){

	int count = 0;
	for (int hh = 0; hh < height_; ++hh){
		for (int ww = 0; ww < width_; ++ww){

			bool occVal = (bool)(occMask_[view].at<uchar>(hh, ww));

			// right
			if (ww + 1 <= width_ - 1) {
				bool occVal_r = (bool)(occMask_[view].at<uchar>(hh, ww + 1));
				if (occVal != occVal_r){
					count++;
				}
			}

			// below
			if (hh + 1 <= height_ - 1) {
				int below_idx = ww + width_ * (hh + 1);
				bool occVal_b = (bool)(occMask_[view].at<uchar>(hh + 1, ww));
				if (occVal != occVal_b){
					count++;
				}

				// left below 
				if (ww - 1 >= 0) {
					int leftBelow_idx = (ww - 1) + width_ * (hh + 1);
					bool occVal_lb = (bool)(occMask_[view].at<uchar>(hh + 1, ww - 1));
					if (occVal != occVal_lb){
						count++;
					}
				}

				// right below
				if (ww + 1 <= width_ - 1) {
					int rightBelow_idx = (ww + 1) + width_ * (hh + 1);
					bool occVal_rb = (bool)(occMask_[view].at<uchar>(hh + 1, ww + 1));
					if (occVal != occVal_rb){
						count++;
					}
				}
			}
		}
	}

	return params_.lambda_occPott * count;
}

void OptModule::printOverallEnergy(OptView view) {

	if (params_.verbose){

		float eData = getEnergyDataTerm(view);
		float ePairwise = getEnergyPairwiseTerm(view);
		float eConsistency = getEnergyConsistencyTerm(view);
		float eSymm = getEnergySymmetryTerm(view);
		float eOccPairwise = getEnergyOccPairwiseTerm(view);

		if (params_.bidirectional == true){
			float allEnergy = eData + ePairwise + eConsistency + eSymm + eOccPairwise;
			//std::cout << "  Total Energy: " << allEnergy << std::endl;
			log_file_ << std::endl << "  - Data term: " << eData << std::endl;
			log_file_ << "  - Pairwise term: " << ePairwise << std::endl;
			log_file_ << "  - Consistency term: " << eConsistency << std::endl;
			log_file_ << "  - Symmetry term: " << eSymm << std::endl;
			log_file_ << "  - Occ Pairwise term: " << eOccPairwise << std::endl;
			log_file_ << "  Total Energy: " << allEnergy << std::endl;
		}
		else{
			float allEnergy = eData + ePairwise;
			//std::cout << "  Total Energy: " << allEnergy << std::endl;
			log_file_ << "  - Data term: " << eData << std::endl;
			log_file_ << "  - Pairwise term: " << ePairwise << std::endl;
			log_file_ << "  Total Energy: " << allEnergy << std::endl;
		}

	}
}

cv::Mat OptModule::visualizeFlowMap(OptView view) {

	cv::Mat output = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat uFloat = cv::Mat(height_, width_, CV_32F, cv::Scalar(0));
	cv::Mat vFloat = cv::Mat(height_, width_, CV_32F, cv::Scalar(0));

	float maxMag = 0.f;

	for (int nn = 0; nn < (int)segments_[view]->size(); ++nn) {

		cv::Mat state = segments_[view]->at(nn).getState();

		// semanticFlow
		std::vector<cv::Point2f> srcPoints, dstPoints;
		srcPoints = segments_[view]->at(nn).getPixelVector();

		getTransformedPoints(srcPoints, state, dstPoints);

		for (int pp = 0; pp < (int)srcPoints.size(); ++pp) {

			float dx = dstPoints[pp].x - srcPoints[pp].x;
			float dy = dstPoints[pp].y - srcPoints[pp].y;

			uFloat.at<float>(cv::Point((int)srcPoints[pp].x, (int)srcPoints[pp].y)) = dx;
			vFloat.at<float>(cv::Point((int)srcPoints[pp].x, (int)srcPoints[pp].y)) = dy;

			float mag = sqrtf(dx*dx + dy*dy);
			if (mag > maxMag) {
				maxMag = mag;
			}
		}
	}

	float maxFlow = std::min(250.f, maxMag);
	//float maxFlow = 120; // sintel

	for (int hh = 0; hh < height_; ++hh) {
		for (int ww = 0; ww < width_; ww++) {

			float dx = uFloat.at<float>(cv::Point(ww, hh));
			float dy = vFloat.at<float>(cv::Point(ww, hh));

			unsigned char pix[3];

			computeColor(dx / maxFlow, dy / maxFlow, pix);

			output.at<cv::Vec3b>(cv::Point(ww, hh)) = cv::Vec3b(pix[0], pix[1], pix[2]);

		}
	}

	return output;
}

void OptModule::WriteFlowField(const std::string& filename_f, const std::string& filename_b, const std::string& filename_occ_f, const std::string& filename_occ_b) {

	{
		OptView view = currFrame;
		png::image< png::rgb_pixel_16 > image(width_, height_);
		for (int nn = 0; nn < (int)segments_[view]->size(); ++nn){
			cv::Mat state = segments_[view]->at(nn).getState();
			std::vector<cv::Point2f> srcPoints, dstPoints;
			srcPoints = segments_[view]->at(nn).getPixelVector();
			getTransformedPoints(srcPoints, state, dstPoints);

			for (int pp = 0; pp < (int)srcPoints.size(); ++pp){

				float dx = dstPoints[pp].x - srcPoints[pp].x;
				float dy = dstPoints[pp].y - srcPoints[pp].y;

				// code from kitti dev-kit
				png::rgb_pixel_16 val;
				val.red = 0;
				val.green = 0;
				val.blue = 0;

				val.red = (uint16_t)std::max(std::min(dx*64.0f + 32768.0f, 65535.0f), 0.0f);
				val.green = (uint16_t)std::max(std::min(dy*64.0f + 32768.0f, 65535.0f), 0.0f);
				val.blue = 1;

				image.set_pixel((int)srcPoints[pp].x, (int)srcPoints[pp].y, val);
			}
		}
		image.write(filename_f);
	}

	{
		OptView view = nextFrame;
		png::image< png::rgb_pixel_16 > image(width_, height_);
		for (int nn = 0; nn < (int)segments_[view]->size(); ++nn){
			cv::Mat state = segments_[view]->at(nn).getState();
			std::vector<cv::Point2f> srcPoints, dstPoints;
			srcPoints = segments_[view]->at(nn).getPixelVector();
			getTransformedPoints(srcPoints, state, dstPoints);

			for (int pp = 0; pp < (int)srcPoints.size(); ++pp){

				float dx = dstPoints[pp].x - srcPoints[pp].x;
				float dy = dstPoints[pp].y - srcPoints[pp].y;

				// code from kitti dev-kit
				png::rgb_pixel_16 val;
				val.red = 0;
				val.green = 0;
				val.blue = 0;

				val.red = (uint16_t)std::max(std::min(dx*64.0f + 32768.0f, 65535.0f), 0.0f);
				val.green = (uint16_t)std::max(std::min(dy*64.0f + 32768.0f, 65535.0f), 0.0f);
				val.blue = 1;

				image.set_pixel((int)srcPoints[pp].x, (int)srcPoints[pp].y, val);
			}
		}
		image.write(filename_b);
	}

	cv::Mat occ_f = occMask_[currFrame];
	cv::Mat occ_b = occMask_[nextFrame];

	for (int hh = 0; hh < height_; ++hh)
	for (int ww = 0; ww < width_; ++ww){
		int occ_ff = occ_f.at<uchar>(hh, ww);
		if (occ_ff == 0){
			occ_f.at<uchar>(hh, ww) = (uchar)255;
		}
		else{
			occ_f.at<uchar>(hh, ww) = (uchar)0;
		}

		int occ_bb = occ_b.at<uchar>(hh, ww); 
		if (occ_bb == 0){
			occ_b.at<uchar>(hh, ww) = (uchar)255;
		}
		else{
			occ_b.at<uchar>(hh, ww) = (uchar)0;
		}
	}
	cv::imwrite(filename_occ_f, occ_f);
	cv::imwrite(filename_occ_b, occ_b);
}

void OptModule::ReadFlowField(const std::string& file_name, cv::Mat& flowX, cv::Mat& flowY){

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

void OptModule::FreeMemory(){

}