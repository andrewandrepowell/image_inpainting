#include "shared_library.h"
#include "../inpainting_algorithms/inpainting_algorithms.h"

using namespace cv;
using namespace inpainting_algorithms;

static void convert_mat2cv_0(float* input_array,int rows,int cols,Mat& output_array);
static void convert_mat2cv_1(float* input_array,int rows,int cols,Mat& output_array);
static void convert_cv2mat_0(Mat& input_array,float* output_array);

void perform_cahn_hilliard_gillette_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,
	int* total_iters,float* epsilons,int total_stages) {
		Mat cv_input_array;
		Mat cv_mask_array;
		Mat cv_output_array;
		convert_mat2cv_0(input_array,rows,cols,cv_input_array);
		convert_mat2cv_1(mask_array,rows,cols,cv_mask_array);
		perform_cahn_hilliard_gillette_inpainting_0(
			cv_input_array,cv_mask_array,cv_output_array,
			total_iters,epsilons,total_stages);
		convert_cv2mat_0(cv_output_array,output_array);
}

int perform_proposed_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,
	int patch_size_0,int distance_size,
	int* skip_factor,
	float* cahn_epsilons,int* cahn_total_iters,int total_stages,
	char* display_name) {
		Mat cv_input_array;
		Mat cv_mask_array;
		Mat cv_output_array;
		Size cv_skip_factor(skip_factor[0],skip_factor[1]);
		convert_mat2cv_0(input_array,rows,cols,cv_input_array);
		convert_mat2cv_1(mask_array,rows,cols,cv_mask_array);
		int iters = perform_proposed_inpainting_0(
			cv_input_array,cv_mask_array,cv_output_array,
			patch_size_0,distance_size,
			cv_skip_factor,
			cahn_epsilons,cahn_total_iters,total_stages,
			display_name);
		convert_cv2mat_0(cv_output_array,output_array);
		return iters;
}

void perform_bertalmio_pde_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,
	int* total_iters,int* total_inpaint_iters, int* total_anidiffuse_iters,int total_stages,
	float* delta_ts,float* sensitivities,int diffuse_coef) {
		Mat cv_input_array;
		Mat cv_mask_array;
		Mat cv_output_array;
		convert_mat2cv_0(input_array,rows,cols,cv_input_array);
		convert_mat2cv_1(mask_array,rows,cols,cv_mask_array);
		perform_bertalmio_pde_inpainting_0(
			cv_input_array,cv_mask_array,cv_output_array,
			total_iters,total_inpaint_iters,total_anidiffuse_iters,total_stages,
			delta_ts,sensitivities,diffuse_coef);
		convert_cv2mat_0(cv_output_array,output_array);
}

int perform_exem_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,int patch_size_0) {
		Mat cv_input_array;
		Mat cv_mask_array;
		Mat cv_output_array;
		convert_mat2cv_0(input_array,rows,cols,cv_input_array);
		convert_mat2cv_1(mask_array,rows,cols,cv_mask_array);
		int iters = perform_exem_inpainting_0(cv_input_array,cv_mask_array,cv_output_array,patch_size_0);
		convert_cv2mat_0(cv_output_array,output_array);
		return iters;
}

void convert_mat2cv_0(float* input_array,int rows,int cols,Mat& output_array) {
	output_array = Mat(Size(cols,rows),CV_32FC1);
	for (int row=0;row<rows;row++) {
		float* output_array_ptr = output_array.ptr<float>(row);
		for (int col=0;col<cols;col++) {
			output_array_ptr[col] = input_array[col*rows+row];
		}
	}
}

void convert_mat2cv_1(float* input_array,int rows,int cols,Mat& output_array) {
	output_array = Mat(Size(cols,rows),CV_8U);
	for (int row=0;row<rows;row++) {
		unsigned char* output_array_ptr = output_array.ptr<unsigned char>(row);
		for (int col=0;col<cols;col++) {
			output_array_ptr[col] = (unsigned char)input_array[col*rows+row];
		}
	}
}

void convert_cv2mat_0(Mat& input_array,float* output_array) {
	for (int row=0;row<input_array.rows;row++) {
		float* input_array_ptr = input_array.ptr<float>(row);
		for (int col=0;col<input_array.cols;col++) {
			output_array[col*input_array.rows+row] = input_array_ptr[col];
		}
	}
}