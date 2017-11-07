#pragma once

enum OptView {
	currFrame = 0,
	nextFrame = 1
};

inline OptView OtherView(OptView view) {
	if (view == currFrame)
		return nextFrame;
	else
		return currFrame;
}

struct OptParameters {

	std::string prefix = "000000";

	// verbose
	bool verbose = false;
	bool CNN_feature_Extract = false;

	// Region variable
	float region_width = 100.f;
	float region_height = 100.f;
	bool fix_boundary_region = true;
	int n_intervals = 4;
	int superpixelNum = 1000;
	float region_overlap_ratio = 0.5f;

	// Opt
	int n_outer_iterations = 3;
	int n_random_states = 6;
	int n_random_states_all = 30;
	float size_perturb_scale = 4.f;
	float size_perturb_trans = 10.f;

	// bidirectional
	bool bidirectional = false;

	// Energy
	float alpha = 0.88f;
	
	float lambda_pairwise = 0.5f;
	float lambda_occPott = 2.f;
	float lambda_consistency = 0.3f;
	float lambda_symmetry = 4.f;
	float lambda_occp = 20.f;
	
	float truncate_ternary = 22.f;
	float truncate_grad = 16.3f;
	float truncate_data = 21.8f;
	
	float weight_coPlanar = 1.0f;
	float truncate_coPlanar = 2.5f;
	float truncate_pairwise = 15.f;
	float truncate_consistency = 20.f;
	
	float bPixelWeight = 0.015f;
	float superpixelWeight = 0.02f;

	float patch_size = 1.f;
	float truncate_CNNF = 3.5f;

	std::string dirRes = "";
};
