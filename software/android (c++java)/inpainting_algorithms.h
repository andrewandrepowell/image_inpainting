#ifndef INPAINTING_ALGORITHMS_H_
#define INPAINTING_ALGORITHMS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <complex>
#include <limits>

#define IA_ENABLE_HIGH_GUI

namespace inpainting_algorithms {
	
	void perform_cahn_hilliard_gillette_inpainting_0(
		cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
		int* total_iters,float* epsilons,int total_stages);

	void perform_cahn_hilliard_gillette_inpainting_1(
		cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
		int* total_iters,float* epsilons,int total_stages);

	int perform_proposed_inpainting_0(
		cv::Mat& input_array_0,cv::Mat& mask_array_0,cv::Mat& output_array_0,
		int patch_size_0,int distance_size,
		cv::Size& skip_factor,
		float* cahn_epsilons=NULL,int* cahn_total_iters=NULL,int total_stages=0,
		char* display_name=NULL);

	void perform_bertalmio_pde_inpainting_0(
		cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
		int* total_iters,int* total_inpaint_iters, int* total_anidiffuse_iters,int total_stages,
		float* delta_ts,float* sensitivities,int diffuse_coef);

	inline int perform_exem_inpainting_0(
		cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,int patch_size_0=5) {
		cv::Size skip_factor(1,1);
			 return perform_proposed_inpainting_0(input_array,mask_array,output_array,patch_size_0,0,skip_factor);
	}

	class Inpainting_Base_0 {
	public:
		Inpainting_Base_0() {}
		virtual ~Inpainting_Base_0() {}
		virtual bool perform() { return true; }
		virtual void update() { }
	private:
	};

	class Cahn_hilliard_gillette_inpainting_0 : public Inpainting_Base_0 {
	public:
		Cahn_hilliard_gillette_inpainting_0(
			cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
			int* total_iters,float* epsilons,int total_stages);
		~Cahn_hilliard_gillette_inpainting_0() { }
		bool perform();
		void update();
	private:
		typedef std::complex<float> complex_type;
		typedef cv::Mat_<complex_type> complex_mat_type;
		typedef unsigned char logical_type;
		const float lambda_0;
		const float delta_t;
		const float c2;
		const cv::Size original_size;
		const cv::Size padded_size;
		const cv::Rect select_mask;
		cv::Mat input_array_1;
		cv::Mat mask_array_1;
		cv::Mat output_array_1;
		cv::Mat& output_array;
		complex_mat_type Output_array;
		complex_mat_type lambda_array;
		complex_mat_type w_deriv;
		complex_mat_type term_0;
		complex_mat_type term_1;
		complex_mat_type M;
		complex_mat_type denom;
		int* total_iters;
		float* epsilons;
		int total_stages;
		int stage;
		int total_iter;
		float epsilon;
		float c1;
		int iter;
		int get_next_p2(int x) {
			return (int)pow(2,ceil(log((float)x)/log(2.0f)));
		}
		complex_type get_Laplace(int index,int size) {
			float value = 2.0f*CV_PI*((float)index)/((float)size);
			return  complex_type(cos(value),sin(value));  
		}
		void run();
	};

	class Proposed_inpainting_0 : public Inpainting_Base_0 {
	public:
		Proposed_inpainting_0(cv::Mat& input_array_0,cv::Mat& mask_array_0,cv::Mat& output_array_0,
			int patch_size_0,int distance_size,
			cv::Size& skip_factor);
		~Proposed_inpainting_0() { }
		bool perform();
		void update() { output_array = output_array_0(rect_mask_0); }
		int get_iters() { return iters; }
		int get_pixels() { return pixels; }
	private:
		typedef unsigned char logical_type;
		static const int fill_front_size = 3;
		static const int fill_front_area = fill_front_size*fill_front_size;
		const float alpha;
		const int patch_area;
		const int patch_offset_value;
		const int row_start;
		const int col_start;
		const int row_end;
		const int col_end;
		const int row_skip;
		const int col_skip;
		const cv::Point patch_offset;
		const cv::Size patch_size;
		const cv::Size padded_image_size;
		const cv::Rect rect_mask_0;
		int iters;
		int pixels;
		int distance_size;
		cv::Mat fill_front_mask;
		cv::Mat erode_mask;
		cv::Mat border_array;
		cv::Mat confidence_pixel;
		cv::Mat mask_array_1;
		cv::Mat output_array_0;
		cv::Mat& output_array;
		void run();
	};
	
};

#endif
