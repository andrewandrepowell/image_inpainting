#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ios>
#include <limits>
#include <exception>
#include "inpainting_algorithms.h"


using namespace inpainting_algorithms;
using namespace std;

int main(int argc, char* argv[]) {

	const char* image_name = "simple_image_0.png";
	const char* mask_name = "simple_image_0_mask.png";
	const char* window_name = image_name;
	const char* mask_window_name = mask_name;
	const char* output_window_name = "output_array";
	cv::Mat image_array,mask_array,output_array;

	/* Load and normalize the image */
    image_array = cv::imread(image_name,CV_LOAD_IMAGE_GRAYSCALE);  
	image_array.convertTo(image_array,CV_32FC1); 
	cv::normalize(image_array,image_array,0,1,cv::NORM_MINMAX,CV_32FC1);

	/* Load the mask */
	mask_array = cv::imread(mask_name,CV_LOAD_IMAGE_GRAYSCALE);  

	/* Create output_array */
	output_array.create(image_array.size(),CV_32FC1);

	/* Create the windows */
	cv::namedWindow(window_name,cv::WINDOW_AUTOSIZE);
	cv::namedWindow(mask_window_name,cv::WINDOW_AUTOSIZE);
	cv::namedWindow(output_window_name,cv::WINDOW_AUTOSIZE);

	/* Perform the inpainting */

	//int total_iters[] = {1000,10000,10000};
	//float epsilons[] = {3,2,1};
	//int total_stages = 3;
	//perform_cahn_hilliard_gillette_inpainting_0(
	//	image_array,mask_array,output_array,
	//	total_iters,epsilons,total_stages);

	//int total_iters[] = {20000,20000,20000};
	//float epsilons[] = {2,1,.5};
	//int total_stages = 3;
	//perform_cahn_hilliard_gillette_inpainting_1(
	//	image_array,mask_array,output_array,
	//	total_iters,epsilons,total_stages);

	//int total_iters[] = {20000,20000,20000};
	//float epsilons[] = {2,1,.5};
	//int total_stages = 3;
	//Cahn_hilliard_gillette_inpainting_0 cahn_hilliard_obj(
	//		image_array,mask_array,output_array,
	//		total_iters,epsilons,total_stages);
	//while (cahn_hilliard_obj.perform()==false) {
	//	cahn_hilliard_obj.update();
	//	cv::imshow(output_window_name,output_array); 
	//	cv::waitKey(1);  
	//}
	//cahn_hilliard_obj.update();

	//int patch_size = 7;
	//int distance_size = 1000;
	//cv::Size skip_factor(1,1);
	//float cahn_epsilons[] = {1.0f};
	//int cahn_total_iters[] = {100};
	//int total_stages = 1;
	//perform_proposed_inpainting_0(
	//	image_array,mask_array,output_array,
	//	patch_size,distance_size,skip_factor,
	//	cahn_epsilons,cahn_total_iters,total_stages);

	int patch_size = 7;
	int distance_size = 1000;
	cv::Size skip_factor(1,1);
	Proposed_inpainting_0 proposed_inpainting_0_obj(
		image_array,mask_array,output_array,
		patch_size,distance_size,skip_factor);
	while (proposed_inpainting_0_obj.perform()==false) {
		proposed_inpainting_0_obj.update();
		cv::imshow(output_window_name,output_array); 
	 	cv::waitKey(1);  
	}

	//int total_iters[] = {500};
	//int total_inpaint_iters[] = {6};
	//int total_anidiffuse_iters[] = {6};
	//int total_stages = 2;
	//float delta_ts[] = {0.02f};
	//float sensitivites[] = {100};
	//int diffuse_coef = 1;
	//perform_bertalmio_pde_inpainting_0(
	//	image_array,mask_array,output_array,
	//	total_iters,total_inpaint_iters,total_anidiffuse_iters,total_stages,
	//	delta_ts,sensitivites,diffuse_coef);

	/* Display the images */
	cv::imshow(window_name,image_array);  
	cv::imshow(mask_window_name,mask_array); 
	cv::imshow(output_window_name,output_array); 
	cv::waitKey(0);    

	/* Prevent program from closing, immediately */
	cin.ignore(numeric_limits<streamsize>::max(), '\n');
	return 0;
}