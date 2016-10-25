#include "inpainting_algorithms.h"
#include <iostream>

namespace inpainting_algorithms {

	void perform_cahn_hilliard_gillette_inpainting_0(
		cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
		int* total_iters,float* epsilons,int total_stages) {

			/* Declarations */
			typedef std::complex<float> complex_type;
			typedef cv::Mat_<complex_type> complex_mat_type;
			typedef unsigned char logical_type;
			auto get_next_p2 = [](int x) { 
				return (int)pow(2,ceil(log((float)x)/log(2.0f))); 
			};
			const float lambda_0 = 1.0f;
			const float delta_t = 1;
			const float c2 = lambda_0*100;
			const cv::Size original_size(get_next_p2(input_array.cols),get_next_p2(input_array.rows));

			/* Select mask */
			cv::Rect select_rect(0,0,input_array.cols,input_array.rows);

			/* Matrix declarations */
			complex_mat_type original_array(original_size,complex_type(0,0));
			complex_mat_type update_array(original_size,complex_type(0,0));
			complex_mat_type lambda_array(original_size,complex_type(0,0));
			complex_mat_type w_deriv(original_size,complex_type(0,0));
			complex_mat_type term_0(original_size,complex_type(0,0));
			complex_mat_type term_1(original_size,complex_type(0,0));
			complex_mat_type M(original_size,complex_type(0,0));
			complex_mat_type denom(original_size,complex_type(0,0));

			/* Autonomous functions needed for the initialization of the laplace operator in the frequency domain */
			auto get_Laplace = [](int index,int size) { 
				float value = 2.0f*CV_PI*((float)index)/((float)size);
				return  complex_type(cos(value),sin(value));  
			};

			/* Initialize output array */
			if (output_array.empty()) {
				output_array = cv::Mat(input_array.size(),CV_32FC1);
			}

			/* Initialize arrays with input_array */
			for (int row=0;row<output_array.rows;row++) {
				logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
				complex_type* lambda_array_ptr = lambda_array.ptr<complex_type>(row);
				float* input_array_ptr = input_array.ptr<float>(row);
				complex_type* original_array_ptr = original_array.ptr<complex_type>(row);
				complex_type* update_array_ptr = update_array.ptr<complex_type>(row);
				for (int col=0;col<output_array.cols;col++) {

					/* Set lambda_array */
					if (mask_array_ptr[col]==0) {
						lambda_array_ptr[col] = lambda_0;
					}

					/* Set other arrays */
					original_array_ptr[col] = input_array_ptr[col];
					update_array_ptr[col] = input_array_ptr[col];
				}
			}

			/* Initialize values, element by element */
			for (int row=0;row<M.rows;row++) {
				complex_type* M_ptr = M.ptr<complex_type>(row);
				for (int col=0;col<M.cols;col++) {
					
					/* Initialize the Laplace matrix M */
					int row_0 = row+1, col_0 = col+1;
					M_ptr[col] = 
						get_Laplace(-row_0,M.rows)+get_Laplace(row_0,M.rows)+
						get_Laplace(-col_0,M.cols)+get_Laplace(col_0,M.cols)-4.0f;
				}
			}

			/* Run algorithm for each stage */
			for (int stage=0;stage<total_stages;stage++) {

				/* Gather data */
				int total_iter = total_iters[stage];
				float epsilon = epsilons[stage];
				float c1 = 1.0f/epsilon; 

				/* Determine denominator */
				for (int row=0;row<M.rows;row++) {
					complex_type* M_ptr = M.ptr<complex_type>(row);
					complex_type* denom_ptr = denom.ptr<complex_type>(row);
					for (int col=0;col<M.cols;col++) {
						denom_ptr[col] = 1.0f+epsilon*M_ptr[col]*M_ptr[col]-c1*M_ptr[col]+c2;
					}
				}

				/* Perform Cahn-Hilliard algorithm */
				for (int iter=0;iter<total_iter;iter++) {
					
					/* Perform some element-by-element calculations */
					for (int row=0;row<M.rows;row++) {
						complex_type* update_array_ptr = update_array.ptr<complex_type>(row);
						complex_type* lambda_array_ptr = lambda_array.ptr<complex_type>(row);
						complex_type* original_array_ptr = original_array.ptr<complex_type>(row);
						complex_type* w_deriv_ptr = w_deriv.ptr<complex_type>(row);
						complex_type* term_1_ptr = term_1.ptr<complex_type>(row);
						for (int col=0;col<M.cols;col++) {

							/* Make sure the imaginary components are set to zero */
							update_array_ptr[col] = update_array_ptr[col].real();

							/* Derive w_deriv */
							w_deriv_ptr[col] = 
								4.0f*update_array_ptr[col]*update_array_ptr[col]*update_array_ptr[col]-
								6.0f*update_array_ptr[col]*update_array_ptr[col]+
								2.0f*update_array_ptr[col];

							/* Derive term_1 */
							term_1_ptr[col] = 
								lambda_array_ptr[col]*(original_array_ptr[col])+
								(c2-lambda_array_ptr[col])*update_array_ptr[col];
						}
					}

					/* Perform transforms */
					dft(w_deriv,w_deriv);
					dft(term_1,term_1);
					dft(update_array,update_array);

					/* Perform some element-by-element calculations */
					for (int row=0;row<M.rows;row++) {
						complex_type* update_array_ptr = update_array.ptr<complex_type>(row);
						complex_type* M_ptr = M.ptr<complex_type>(row);
						complex_type* w_deriv_ptr = w_deriv.ptr<complex_type>(row);
						complex_type* term_0_ptr = term_0.ptr<complex_type>(row);
						complex_type* term_1_ptr = term_1.ptr<complex_type>(row);
						complex_type* denom_ptr = denom.ptr<complex_type>(row);
						for (int col=0;col<M.cols;col++) {

							/* Derive term_0 */
							term_0_ptr[col] =
								M_ptr[col]*(w_deriv_ptr[col]/epsilon-c1*update_array_ptr[col]);

							/* Derive update_array */
							update_array_ptr[col] = 
								(delta_t*(term_0_ptr[col]+term_1_ptr[col])+update_array_ptr[col])/denom_ptr[col];
						}
					}

					/* Acquire update_array by going back to the temporal domain */
					dft(update_array,update_array,cv::DFT_INVERSE|cv::DFT_SCALE);
				}
			}

			/* Copy results when finised */
			for (int row=0;row<output_array.rows;row++) {
				float* output_array_ptr = output_array.ptr<float>(row);
				float* input_array_ptr = input_array.ptr<float>(row);
				logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
				complex_type* update_array_ptr = update_array.ptr<complex_type>(row);
				for (int col=0;col<output_array.cols;col++) {
					output_array_ptr[col] = (mask_array_ptr[col]!=0) ? 
						update_array_ptr[col].real() : input_array_ptr[col];
				}
			}
	}

	void perform_cahn_hilliard_gillette_inpainting_1(
		cv::Mat& input_array_0,cv::Mat& mask_array_0,cv::Mat& output_array,
		int* total_iters,float* epsilons,int total_stages) {

			/* Declarations */
			typedef std::complex<float> complex_type;
			typedef cv::Mat_<complex_type> complex_mat_type;
			typedef unsigned char logical_type;
			auto get_next_p2 = [](int x) { return (int)pow(2,ceil(log((float)x)/log(2.0f))); };
			const float lambda_0 = 1.0f;
			const float delta_t = 1;
			const float c2 = lambda_0*100;
			const cv::Size original_size = input_array_0.size();
			const cv::Size padded_size(get_next_p2(input_array_0.cols),get_next_p2(input_array_0.rows));
			const cv::Rect select_mask(cv::Point(0,0),original_size);

			/* Matrix declarations */
			cv::Mat mask_array_1,input_array_1;
			complex_mat_type Output_array(padded_size,complex_type(0,0));
			complex_mat_type lambda_array(padded_size,complex_type(0,0));
			complex_mat_type w_deriv(padded_size,complex_type(0,0));
			complex_mat_type term_0(padded_size,complex_type(0,0));
			complex_mat_type term_1(padded_size,complex_type(0,0));
			complex_mat_type M(padded_size,complex_type(0,0));
			complex_mat_type denom(padded_size,complex_type(0,0));

			/* Autonomous functions needed for the initialization of the laplace operator in the frequency domain */
			auto get_Laplace = [](int index,int size) { 
				float value = 2.0f*CV_PI*((float)index)/((float)size);
				return  complex_type(cos(value),sin(value));  
			};

			/* Initialize arrays with input_array */
			cv::copyMakeBorder(input_array_0,input_array_1,
				0,padded_size.height-original_size.height,0,padded_size.width-original_size.width,
				cv::BORDER_CONSTANT,0);
			cv::copyMakeBorder(mask_array_0,mask_array_1,
				0,padded_size.height-original_size.height,0,padded_size.width-original_size.width,
				cv::BORDER_CONSTANT,0);
			input_array_1.copyTo(output_array);
			lambda_array.setTo(cv::Scalar::all(lambda_0),mask_array_1==0);

			/* Initialize values, element by element */
			for (int row=0;row<M.rows;row++) {
				complex_type* M_ptr = M.ptr<complex_type>(row);
				for (int col=0;col<M.cols;col++) {
					
					/* Initialize the Laplace matrix M */
					int row_0 = row+1, col_0 = col+1;
					M_ptr[col] = 
						get_Laplace(-row_0,M.rows)+get_Laplace(row_0,M.rows)+
						get_Laplace(-col_0,M.cols)+get_Laplace(col_0,M.cols)-4.0f;
				}
			}

			/* Run algorithm for each stage */
			for (int stage=0;stage<total_stages;stage++) {

				/* Gather data */
				int total_iter = total_iters[stage];
				float epsilon = epsilons[stage];
				float c1 = 1.0f/epsilon; 

				/* Determine denominator */
				for (int row=0;row<M.rows;row++) {
					complex_type* M_ptr = M.ptr<complex_type>(row);
					complex_type* denom_ptr = denom.ptr<complex_type>(row);
					for (int col=0;col<M.cols;col++) {
						denom_ptr[col] = 1.0f+epsilon*M_ptr[col]*M_ptr[col]-c1*M_ptr[col]+c2;
					}
				}

				/* Perform Cahn-Hilliard algorithm */
				for (int iter=0;iter<total_iter;iter++) {
					
					/* Perform some element-by-element calculations */
					for (int row=0;row<M.rows;row++) {
						float* output_array_ptr = output_array.ptr<float>(row);
						float* input_array_1_ptr = input_array_1.ptr<float>(row);
						complex_type* lambda_array_ptr = lambda_array.ptr<complex_type>(row);
						complex_type* w_deriv_ptr = w_deriv.ptr<complex_type>(row);
						complex_type* term_1_ptr = term_1.ptr<complex_type>(row);
						for (int col=0;col<M.cols;col++) {

							/* Derive w_deriv */
							w_deriv_ptr[col] = 
								4.0f*output_array_ptr[col]*output_array_ptr[col]*output_array_ptr[col]-
								6.0f*output_array_ptr[col]*output_array_ptr[col]+
								2.0f*output_array_ptr[col];

							/* Derive term_1 */
							term_1_ptr[col] = 
								lambda_array_ptr[col]*(input_array_1_ptr[col])+
								(c2-lambda_array_ptr[col])*output_array_ptr[col];
						}
					}

					/* Perform transforms */
					dft(w_deriv,w_deriv);
					dft(term_1,term_1);
					dft(output_array,Output_array,cv::DFT_COMPLEX_OUTPUT);

					/* Perform some element-by-element calculations */
					for (int row=0;row<M.rows;row++) {
						complex_type* Output_array_ptr = Output_array.ptr<complex_type>(row);
						complex_type* M_ptr = M.ptr<complex_type>(row);
						complex_type* w_deriv_ptr = w_deriv.ptr<complex_type>(row);
						complex_type* term_0_ptr = term_0.ptr<complex_type>(row);
						complex_type* term_1_ptr = term_1.ptr<complex_type>(row);
						complex_type* denom_ptr = denom.ptr<complex_type>(row);
						for (int col=0;col<M.cols;col++) {

							/* Derive term_0 */
							term_0_ptr[col] =
								M_ptr[col]*(w_deriv_ptr[col]/epsilon-c1*Output_array_ptr[col]);

							/* Derive update_array */
							Output_array_ptr[col] = 
								(delta_t*(term_0_ptr[col]+term_1_ptr[col])+Output_array_ptr[col])/denom_ptr[col];
						}
					}

					/* Acquire update_array by going back to the temporal domain */
					dft(Output_array,output_array,cv::DFT_INVERSE|cv::DFT_SCALE|cv::DFT_REAL_OUTPUT);
				}
			}

			/* Extract results when finised */
			input_array_1.copyTo(output_array,mask_array_1==0);
			output_array = output_array(select_mask);
	}

	int perform_proposed_inpainting_0(
		cv::Mat& input_array_0,cv::Mat& mask_array_0,cv::Mat& output_array_0,
		int patch_size_0,int distance_size,cv::Size& skip_factor,
		float* cahn_epsilons,int* cahn_total_iters,int total_stages,
		char* display_name) {

			typedef unsigned char logical_type;
			static const float alpha = 1;
			static const int fill_front_size = 3;
			static const int fill_front_area = fill_front_size*fill_front_size;
			const int patch_area = patch_size_0*patch_size_0;
			const int patch_offset_value = patch_size_0>>1;
			const int row_start = patch_offset_value;
			const int col_start = patch_offset_value;
			const int row_end = input_array_0.rows+patch_offset_value;
			const int col_end = input_array_0.cols+patch_offset_value;
			const int row_skip = skip_factor.height;
			const int col_skip = skip_factor.width;
			const cv::Point patch_offset(patch_offset_value,patch_offset_value);
			const cv::Size patch_size(patch_size_0,patch_size_0);
			const cv::Size padded_image_size(col_start+col_end,row_start+row_end);
			int iters = 0;

			cv::Mat fill_front_mask(fill_front_size,fill_front_size,CV_8U,cv::Scalar::all(1));
			cv::Mat erode_mask = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));

			cv::Mat border_array(input_array_0.size(),CV_8U,cv::Scalar::all(1));
			cv::Mat confidence_pixel;
			cv::Mat mask_array_1;

			/* Initialize matrix filled with the confidence values for each pixel */
			confidence_pixel = (mask_array_0==0)&1;
			confidence_pixel.convertTo(confidence_pixel,CV_32FC1);
			cv::copyMakeBorder(confidence_pixel,confidence_pixel,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_REFLECT_101);

			/* Initialize the changing matrix array */
			mask_array_1 = (mask_array_0!=0)&1;
			cv::copyMakeBorder(mask_array_1,mask_array_1,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_CONSTANT,0);

			/* Initialize the output array */
			output_array_0 = input_array_0.clone();
			cv::copyMakeBorder(output_array_0,output_array_0,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_REFLECT_101);

			/* Initialize the border array */
			cv::copyMakeBorder(border_array,border_array,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_CONSTANT,0);

			/* Continue algorithm until there are no more
			   pixels to fill */
			while (cv::countNonZero(mask_array_1!=0)) {
				
				/* Determine the fill front */
				cv::Mat fill_front;
				cv::filter2D(mask_array_1,fill_front,-1,fill_front_mask);
				fill_front = ((fill_front<fill_front_area)&mask_array_1)&1;

				/* Determine the vectors normal to the occlusion */
				cv::Mat n_row,n_col;
				cv::Sobel(mask_array_1,n_row,-1,0,1);
				cv::Sobel(mask_array_1,n_col,-1,1,0);
				n_row.convertTo(n_row,CV_32FC1);
				n_col.convertTo(n_col,CV_32FC1);

				/* Determine the isophotes from the image */
				cv::Mat iso_row,iso_col;
				cv::Sobel(output_array_0,iso_row,-1,1,0);
				cv::Sobel(output_array_0,iso_col,-1,0,1);
				iso_row *= -1;

				/* Determine patch isophotes and confidences along the fill front */
				cv::Mat isophote_row_patch(padded_image_size,CV_32FC1,cv::Scalar::all(0));
				cv::Mat isophote_col_patch(padded_image_size,CV_32FC1,cv::Scalar::all(0));
				cv::Mat confidence_patch(padded_image_size,CV_32FC1,cv::Scalar::all(0));
				for (int row=row_start;row<row_end;row++) {
					logical_type* fill_front_ptr = fill_front.ptr<logical_type>(row);
					float* confidence_patch_ptr = confidence_patch.ptr<float>(row);
					float* isophote_row_patch_ptr = isophote_row_patch.ptr<float>(row);
					float* isophote_col_patch_ptr = isophote_col_patch.ptr<float>(row);
					for (int col=col_start;col<col_end;col++) {
						if (fill_front_ptr[col]!=0) {

							/* Determine location */
							cv::Rect rect_mask_0(cv::Point(col,row)-patch_offset,patch_size);

							/* Determine confidences of each patch */
							cv::Mat confidence_patch_0 = confidence_pixel(rect_mask_0);
							cv::Mat confidence_mask_0 = (mask_array_1(rect_mask_0)==0)&1;
							confidence_mask_0.convertTo(confidence_mask_0,CV_32FC1);
							confidence_patch_ptr[col] = cv::sum(confidence_patch_0.mul(confidence_mask_0))[0]/patch_area;

							/* Determine isophote patch */
							cv::Mat iso_row_0 = iso_row(rect_mask_0);
							cv::Mat iso_col_0 = iso_col(rect_mask_0);
							cv::Mat iso_mask_0;
							cv::erode(confidence_mask_0,iso_mask_0,erode_mask);

							/* The average isophote of the patch, exclusing those in the occlusion, is the isophote of the patch */
							float iso_total_0 = cv::sum(iso_mask_0)[0];
							if (iso_total_0!=0) {
								isophote_row_patch_ptr[col] = cv::sum(iso_row_0.mul(iso_mask_0))[0]/iso_total_0;
								isophote_col_patch_ptr[col] = cv::sum(iso_col_0.mul(iso_mask_0))[0]/iso_total_0;
							} else {
								isophote_row_patch_ptr[col] = 0;
								isophote_col_patch_ptr[col] = 0;
							}
						}
					}
				}
				
				/* Determine data_patch and priority patch */
				cv::Mat data_patch = cv::abs(isophote_row_patch.mul(n_row)+isophote_col_patch.mul(n_col))/alpha;
				cv::Mat priority_patch = data_patch.mul(confidence_patch);

				/* Determine largest priority's location on fill front */
				cv::Point p_patch_loc;
				cv::minMaxLoc(priority_patch,NULL,NULL,NULL,&p_patch_loc,fill_front);

				/* Also, determine p patch's mask and values */
				cv::Rect p_patch_rect_mask(p_patch_loc-patch_offset,patch_size);
				cv::Mat p_patch_mask = (mask_array_1(p_patch_rect_mask)==0)&1;
				cv::Mat p_patch_valu = output_array_0(p_patch_rect_mask);
				p_patch_mask.convertTo(p_patch_mask,CV_32FC1);

				/* Determine the mse for every filled patch */
				cv::Mat mse_patch(padded_image_size,CV_32FC1,
					cv::Scalar::all((std::numeric_limits<float>::infinity())));
				for (int row=row_start;row<row_end;row+=row_skip) {
					float* mse_patch_ptr = mse_patch.ptr<float>(row);
					for (int col=col_start;col<col_end;col+=col_skip) {
	
						/* Determine q patch location */
						cv::Point q_patch_loc(col,row);
						cv::Rect q_patch_rect_mask(q_patch_loc-patch_offset,patch_size);
						cv::Mat q_patch_mask = (mask_array_1(q_patch_rect_mask)==0)&1;

						/* Determine distance between q and p */
						int x_diff = q_patch_loc.x-p_patch_loc.x;
						int y_diff = q_patch_loc.y-p_patch_loc.y;
						
						/* Only generate the q patch's mse if the q-patch is not in the occlusion and
						   is within the specified distance, that is, distance_size */
						bool flag_0 = (cv::countNonZero(q_patch_mask)==patch_area);
						bool flag_1 = (distance_size>0)?(((x_diff*x_diff)+(y_diff*y_diff))<=(distance_size*distance_size)):true;
						if (flag_0&&flag_1) {

								/* Determine the mse */
								cv::Mat q_patch_valu = output_array_0(q_patch_rect_mask);
								mse_patch_ptr[col] = cv::sum(cv::abs(q_patch_valu-p_patch_valu).mul(p_patch_mask))[0];
						}
					}
				}
				
				/* Determine the replacement location which is defined by the minimum mse in the area that is
				   not the occlusion */
				cv::Point r_patch_loc;
				cv::minMaxLoc(mse_patch,NULL,NULL,&r_patch_loc,NULL,(mask_array_1==0)&border_array);

				/* Determine the replacement patch's values and mask */
				cv::Rect r_patch_rect_mask(r_patch_loc-patch_offset,patch_size);
				cv::Mat r_patch_valu = output_array_0(r_patch_rect_mask);
				cv::Mat r_patch_mask = (mask_array_1(p_patch_rect_mask)!=0)&1;

				/* Also, let's set the mask values to 0 to indicate the pixels
				   have been filled and set the confidence pixel values to that
				   of the patch */
				cv::Mat m_patch_valu = mask_array_1(p_patch_rect_mask);
				cv::Mat c_patch_valu = confidence_pixel(p_patch_rect_mask);
				float p_patch_confid = confidence_patch.at<float>(p_patch_loc);

				/* Perform replacement */
				for (int row=0;row<p_patch_valu.rows;row++) {
					logical_type* r_patch_mask_ptr = r_patch_mask.ptr<logical_type>(row);
					logical_type* m_patch_valu_ptr = m_patch_valu.ptr<logical_type>(row);
					float* p_patch_valu_ptr = p_patch_valu.ptr<float>(row);
					float* r_patch_valu_ptr = r_patch_valu.ptr<float>(row);
					float* c_patch_valu_ptr = c_patch_valu.ptr<float>(row);
					for (int col=0;col<p_patch_valu.cols;col++) {
						if (r_patch_mask_ptr[col]!=0) {
							p_patch_valu_ptr[col] = r_patch_valu_ptr[col];
							m_patch_valu_ptr[col] = 0;
							c_patch_valu_ptr[col] = p_patch_confid;
						}
					}
				}

				/* If display mode is enable, display the output array */
#ifdef IA_ENABLE_HIGH_GUI
				if (display_name!=NULL) {
					std::cout << "Pixels remaining: " << cv::countNonZero(mask_array_1!=0) << "\t";
					std::cout << "Iterations: " << iters << std::endl;
					cv::imshow(display_name,output_array_0);
					cv::waitKey(1);
				}
#endif

				/* Incrment the total number of iterations */
				iters++;
			}
			
			/* Remove border */
			cv::Rect rect_mask_0(patch_offset,input_array_0.size());
			output_array_0 = output_array_0(rect_mask_0);
			
			/* Apply cahn hilliard */
			if (total_stages!=0) {
				perform_cahn_hilliard_gillette_inpainting_0(
					output_array_0,mask_array_0,output_array_0,
					cahn_total_iters,cahn_epsilons,total_stages);
			}

			/* Return the total number of iterations for determining metric information */
			return iters;
	}

	void perform_bertalmio_pde_inpainting_0(
		cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
		int* total_iters,int* total_inpaint_iters, int* total_anidiffuse_iters,int total_stages,
		float* delta_ts,float* sensitivities,int diffuse_coef) {

			/* Other declarations */
			typedef unsigned char logical_type;

			/* Matrix declarations */
			cv::Mat image_grad_row;
			cv::Mat image_grad_col;
			cv::Mat image_grad_norm;
			cv::Mat image_iso_row;
			cv::Mat image_iso_col;
			cv::Mat image_iso_norm;
			cv::Mat image_laplacian;
			cv::Mat image_laplacian_grad_row;
			cv::Mat image_laplacian_grad_col;
			cv::Mat diffuse_coefs;
			cv::Mat temp;

			/* Initialize output */
			input_array.copyTo(output_array);

			/* Compute bertalmio for each stage */
			for (int stage=0;stage<total_stages;stage++) {

				/* Grab data */
				int total_iter = total_iters[stage];
				int total_inpaint_iter = total_inpaint_iters[stage];
				int total_anidiffuse_iter = total_anidiffuse_iters[stage];
				float sensitivity = sensitivities[stage];
				float delta_t = delta_ts[stage];

				/* Run stage of algorithm */
				for (int iter=0;iter<total_iter;iter++) {

					/* Perform anisotropic diffusion (there's probably a function for this, but wutevs) */
					for (int iter_aniffuse=0;iter_aniffuse<total_anidiffuse_iter;iter_aniffuse++) {
						cv::Sobel(output_array,image_grad_row,-1,0,1);
						cv::Sobel(output_array,image_grad_col,-1,1,0);
						cv::magnitude(image_grad_row,image_grad_col,image_grad_norm);
						if (diffuse_coef==0) {
							cv::exp(-(image_grad_norm.mul(1/sensitivity)),diffuse_coefs);
						} else {
							cv::pow(image_grad_norm.mul(1/sensitivity),2,temp);
							diffuse_coefs = 1/(1+temp);
						}
						cv::Laplacian(output_array,image_laplacian,-1);
						for (int row=0;row<output_array.rows;row++) {
							float* output_array_ptr = output_array.ptr<float>(row);
							float* diffuse_coefs_ptr = diffuse_coefs.ptr<float>(row);
							float* image_laplacian_ptr = image_laplacian.ptr<float>(row);
							logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
							for (int col=0;col<output_array.cols;col++) {
								if (mask_array_ptr[col]!=0) {
									output_array_ptr[col] +=
										delta_t*(diffuse_coefs_ptr[col]*image_laplacian_ptr[col]);
								}
							}
						}
					}

					/* Perform inpainting */
					for (int total_inpaint_iters=0;total_inpaint_iters<total_inpaint_iter;total_inpaint_iters++) {
						cv::Sobel(output_array,image_iso_row,-1,1,0);
						cv::Sobel(output_array,image_iso_col,-1,0,1);
						image_iso_row *= -1;
						cv::sqrt(image_iso_row.mul(image_iso_row)+image_iso_col.mul(image_iso_col),image_iso_norm);
						cv::Laplacian(output_array,image_laplacian,-1);
						cv::Sobel(image_laplacian,image_laplacian_grad_row,-1,0,1);
						cv::Sobel(image_laplacian,image_laplacian_grad_col,-1,1,0);
						for (int row=0;row<output_array.rows;row++) {
							logical_type* mask_array_ptr = mask_array.ptr<logical_type>(row);
							float* image_iso_norm_ptr = image_iso_norm.ptr<float>(row);
							float* image_iso_row_ptr = image_iso_row.ptr<float>(row);
							float* image_iso_col_ptr = image_iso_col.ptr<float>(row);
							float* image_laplacian_grad_row_ptr = image_laplacian_grad_row.ptr<float>(row);
							float* image_laplacian_grad_col_ptr = image_laplacian_grad_col.ptr<float>(row);
							float* output_array_ptr = output_array.ptr<float>(row);
							for (int col=0;col<output_array.cols;col++) {
								if ((mask_array_ptr[col]!=0)&&(image_iso_norm_ptr[col]!=0)) {
									output_array_ptr[col]  -= delta_t*(
										image_iso_row_ptr[col]*image_laplacian_grad_row_ptr[col]+
										image_iso_col_ptr[col]*image_laplacian_grad_col_ptr[col])/
										image_iso_norm_ptr[col];
									//std::cout << image_iso_norm_ptr[col] << std::endl;
									output_array_ptr[col] = (output_array_ptr[col]>1.0f)?1:output_array_ptr[col];
									output_array_ptr[col] = (output_array_ptr[col]<0.0f)?0:output_array_ptr[col];
									//std::cout << output_array_ptr[col] << std::endl;
								}
							}
						}
					}
				}
			}

	}

	Cahn_hilliard_gillette_inpainting_0::Cahn_hilliard_gillette_inpainting_0(
			cv::Mat& input_array,cv::Mat& mask_array,cv::Mat& output_array,
			int* total_iters,float* epsilons,int total_stages) : lambda_0(1.0f), delta_t(1), c2(lambda_0*100),
	output_array(output_array),total_iters(total_iters), epsilons(epsilons), total_stages(total_stages),
		original_size(input_array.size()),padded_size(get_next_p2(input_array.cols),get_next_p2(input_array.rows)),
		select_mask(cv::Point(0,0),original_size),Output_array(padded_size,complex_type(0,0)),lambda_array(padded_size,complex_type(0,0)),
		w_deriv(padded_size,complex_type(0,0)),term_0(padded_size,complex_type(0,0)),term_1(padded_size,complex_type(0,0)),M(padded_size,complex_type(0,0)),
		denom(padded_size,complex_type(0,0)),stage(0),total_iter(total_iters[0]),iter(total_iter) {

			/* Initialize arrays with input_array */
			cv::copyMakeBorder(input_array,input_array_1,
				0,padded_size.height-original_size.height,0,padded_size.width-original_size.width,
				cv::BORDER_CONSTANT,0);
			cv::copyMakeBorder(mask_array,mask_array_1,
				0,padded_size.height-original_size.height,0,padded_size.width-original_size.width,
				cv::BORDER_CONSTANT,0);
			input_array_1.copyTo(output_array_1);
			lambda_array.setTo(cv::Scalar::all(lambda_0),mask_array_1==0);

			/* Initialize values, element by element */
			for (int row=0;row<M.rows;row++) {
				complex_type* M_ptr = M.ptr<complex_type>(row);
				for (int col=0;col<M.cols;col++) {
					
					/* Initialize the Laplace matrix M */
					int row_0 = row+1, col_0 = col+1;
					M_ptr[col] = 
						get_Laplace(-row_0,M.rows)+get_Laplace(row_0,M.rows)+
						get_Laplace(-col_0,M.cols)+get_Laplace(col_0,M.cols)-4.0f;
				}
			}
	}

	bool Cahn_hilliard_gillette_inpainting_0::perform() {

		if (iter==total_iter) {
			if (stage==total_stages) {
				return true;
			} else {
				total_iter = total_iters[stage];
				epsilon = epsilons[stage];
				c1 = 1.0f/epsilon; 
				for (int row=0;row<M.rows;row++) {
					complex_type* M_ptr = M.ptr<complex_type>(row);
					complex_type* denom_ptr = denom.ptr<complex_type>(row);
					for (int col=0;col<M.cols;col++) {
						denom_ptr[col] = 1.0f+epsilon*M_ptr[col]*M_ptr[col]-c1*M_ptr[col]+c2;
					}
				}
				iter = 0;
				run();
				stage++;
			}
		} else {
			run();
			iter++;
		}
		return false;
	}

	void Cahn_hilliard_gillette_inpainting_0::run() {

		/* Perform some element-by-element calculations */
		for (int row=0;row<M.rows;row++) {
			float* output_array_1_ptr = output_array_1.ptr<float>(row);
			float* input_array_1_ptr = input_array_1.ptr<float>(row);
			complex_type* lambda_array_ptr = lambda_array.ptr<complex_type>(row);
			complex_type* w_deriv_ptr = w_deriv.ptr<complex_type>(row);
			complex_type* term_1_ptr = term_1.ptr<complex_type>(row);
			for (int col=0;col<M.cols;col++) {

				/* Derive w_deriv */
				w_deriv_ptr[col] = 
					4.0f*output_array_1_ptr[col]*output_array_1_ptr[col]*output_array_1_ptr[col]-
					6.0f*output_array_1_ptr[col]*output_array_1_ptr[col]+
					2.0f*output_array_1_ptr[col];

				/* Derive term_1 */
				term_1_ptr[col] = 
					lambda_array_ptr[col]*(input_array_1_ptr[col])+
					(c2-lambda_array_ptr[col])*output_array_1_ptr[col];
			}
		}

		/* Perform transforms */
		dft(w_deriv,w_deriv);
		dft(term_1,term_1);
		dft(output_array_1,Output_array,cv::DFT_COMPLEX_OUTPUT);

		/* Perform some element-by-element calculations */
		for (int row=0;row<M.rows;row++) {
			complex_type* Output_array_ptr = Output_array.ptr<complex_type>(row);
			complex_type* M_ptr = M.ptr<complex_type>(row);
			complex_type* w_deriv_ptr = w_deriv.ptr<complex_type>(row);
			complex_type* term_0_ptr = term_0.ptr<complex_type>(row);
			complex_type* term_1_ptr = term_1.ptr<complex_type>(row);
			complex_type* denom_ptr = denom.ptr<complex_type>(row);
			for (int col=0;col<M.cols;col++) {

				/* Derive term_0 */
				term_0_ptr[col] =
					M_ptr[col]*(w_deriv_ptr[col]/epsilon-c1*Output_array_ptr[col]);

				/* Derive update_array */
				Output_array_ptr[col] = 
					(delta_t*(term_0_ptr[col]+term_1_ptr[col])+Output_array_ptr[col])/denom_ptr[col];
			}
		}

		/* Acquire update_array by going back to the temporal domain */
		dft(Output_array,output_array_1,cv::DFT_INVERSE|cv::DFT_SCALE|cv::DFT_REAL_OUTPUT);
	}

	void Cahn_hilliard_gillette_inpainting_0::update() {
		output_array_1.copyTo(output_array);
		input_array_1.copyTo(output_array,mask_array_1==0);
		output_array = output_array(select_mask);
	}

	Proposed_inpainting_0::Proposed_inpainting_0(cv::Mat& input_array_0,cv::Mat& mask_array_0,cv::Mat& output_array,
			int patch_size_0,int distance_size,
			cv::Size& skip_factor) : 
	output_array(output_array), alpha(1), patch_area(patch_size_0*patch_size_0), patch_offset_value(patch_size_0>>1),
		row_start(patch_offset_value),col_start(patch_offset_value),
		row_end(input_array_0.rows+patch_offset_value),col_end(input_array_0.cols+patch_offset_value),
		row_skip(skip_factor.height),col_skip(skip_factor.width), 
		patch_offset(patch_offset_value,patch_offset_value),patch_size(patch_size_0,patch_size_0),
		padded_image_size(col_start+col_end,row_start+row_end),iters(0),distance_size(distance_size),rect_mask_0(patch_offset,input_array_0.size()),
		fill_front_mask(fill_front_size,fill_front_size,CV_8U,cv::Scalar::all(1)),erode_mask(cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3))),
		border_array(input_array_0.size(),CV_8U,cv::Scalar::all(1)) {

			/* Initialize matrix filled with the confidence values for each pixel */
			confidence_pixel = (mask_array_0==0)&1;
			confidence_pixel.convertTo(confidence_pixel,CV_32FC1);
			cv::copyMakeBorder(confidence_pixel,confidence_pixel,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_REFLECT_101);

			/* Initialize the changing matrix array */
			mask_array_1 = (mask_array_0!=0)&1;
			cv::copyMakeBorder(mask_array_1,mask_array_1,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_CONSTANT,0);

			/* Initialize the output array */
			output_array_0 = input_array_0.clone();
			cv::copyMakeBorder(output_array_0,output_array_0,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_REFLECT_101);

			/* Initialize the border array */
			cv::copyMakeBorder(border_array,border_array,
				patch_offset_value,patch_offset_value,patch_offset_value,patch_offset_value,
				cv::BORDER_CONSTANT,0);

			/* Determine the number of pixels in inpainted region */
			pixels = cv::countNonZero(mask_array_1!=0);
	}

	bool Proposed_inpainting_0::perform() {

		if (pixels>0) {
			run();
			pixels = cv::countNonZero(mask_array_1!=0);
			return false;
		} else {
			return true;
		}
	}

	void Proposed_inpainting_0::run() {

		/* Determine the fill front */
		cv::Mat fill_front;
		cv::filter2D(mask_array_1,fill_front,-1,fill_front_mask);
		fill_front = ((fill_front<fill_front_area)&mask_array_1)&1;

		/* Determine the vectors normal to the occlusion */
		cv::Mat n_row,n_col;
		cv::Sobel(mask_array_1,n_row,-1,0,1);
		cv::Sobel(mask_array_1,n_col,-1,1,0);
		n_row.convertTo(n_row,CV_32FC1);
		n_col.convertTo(n_col,CV_32FC1);

		/* Determine the isophotes from the image */
		cv::Mat iso_row,iso_col;
		cv::Sobel(output_array_0,iso_row,-1,1,0);
		cv::Sobel(output_array_0,iso_col,-1,0,1);
		iso_row *= -1;

		/* Determine patch isophotes and confidences along the fill front */
		cv::Mat isophote_row_patch(padded_image_size,CV_32FC1,cv::Scalar::all(0));
		cv::Mat isophote_col_patch(padded_image_size,CV_32FC1,cv::Scalar::all(0));
		cv::Mat confidence_patch(padded_image_size,CV_32FC1,cv::Scalar::all(0));
		for (int row=row_start;row<row_end;row++) {
			logical_type* fill_front_ptr = fill_front.ptr<logical_type>(row);
			float* confidence_patch_ptr = confidence_patch.ptr<float>(row);
			float* isophote_row_patch_ptr = isophote_row_patch.ptr<float>(row);
			float* isophote_col_patch_ptr = isophote_col_patch.ptr<float>(row);
			for (int col=col_start;col<col_end;col++) {
				if (fill_front_ptr[col]!=0) {

					/* Determine location */
					cv::Rect rect_mask_0(cv::Point(col,row)-patch_offset,patch_size);

					/* Determine confidences of each patch */
					cv::Mat confidence_patch_0 = confidence_pixel(rect_mask_0);
					cv::Mat confidence_mask_0 = (mask_array_1(rect_mask_0)==0)&1;
					confidence_mask_0.convertTo(confidence_mask_0,CV_32FC1);
					confidence_patch_ptr[col] = cv::sum(confidence_patch_0.mul(confidence_mask_0))[0]/patch_area;

					/* Determine isophote patch */
					cv::Mat iso_row_0 = iso_row(rect_mask_0);
					cv::Mat iso_col_0 = iso_col(rect_mask_0);
					cv::Mat iso_mask_0;
					cv::erode(confidence_mask_0,iso_mask_0,erode_mask);

					/* The average isophote of the patch, exclusing those in the occlusion, is the isophote of the patch */
					float iso_total_0 = cv::sum(iso_mask_0)[0];
					if (iso_total_0!=0) {
						isophote_row_patch_ptr[col] = cv::sum(iso_row_0.mul(iso_mask_0))[0]/iso_total_0;
						isophote_col_patch_ptr[col] = cv::sum(iso_col_0.mul(iso_mask_0))[0]/iso_total_0;
					} else {
						isophote_row_patch_ptr[col] = 0;
						isophote_col_patch_ptr[col] = 0;
					}
				}
			}
		}
				
		/* Determine data_patch and priority patch */
		cv::Mat data_patch = cv::abs(isophote_row_patch.mul(n_row)+isophote_col_patch.mul(n_col))/alpha;
		cv::Mat priority_patch = data_patch.mul(confidence_patch);

		/* Determine largest priority's location on fill front */
		cv::Point p_patch_loc;
		cv::minMaxLoc(priority_patch,NULL,NULL,NULL,&p_patch_loc,fill_front);

		/* Also, determine p patch's mask and values */
		cv::Rect p_patch_rect_mask(p_patch_loc-patch_offset,patch_size);
		cv::Mat p_patch_mask = (mask_array_1(p_patch_rect_mask)==0)&1;
		cv::Mat p_patch_valu = output_array_0(p_patch_rect_mask);
		p_patch_mask.convertTo(p_patch_mask,CV_32FC1);

		/* Determine the mse for every filled patch */
		cv::Mat mse_patch(padded_image_size,CV_32FC1,
			cv::Scalar::all((std::numeric_limits<float>::infinity())));
		for (int row=row_start;row<row_end;row+=row_skip) {
			float* mse_patch_ptr = mse_patch.ptr<float>(row);
			for (int col=col_start;col<col_end;col+=col_skip) {
	
				/* Determine q patch location */
				cv::Point q_patch_loc(col,row);
				cv::Rect q_patch_rect_mask(q_patch_loc-patch_offset,patch_size);
				cv::Mat q_patch_mask = (mask_array_1(q_patch_rect_mask)==0)&1;

				/* Determine distance between q and p */
				int x_diff = q_patch_loc.x-p_patch_loc.x;
				int y_diff = q_patch_loc.y-p_patch_loc.y;
						
				/* Only generate the q patch's mse if the q-patch is not in the occlusion and
					is within the specified distance, that is, distance_size */
				bool flag_0 = (cv::countNonZero(q_patch_mask)==patch_area);
				bool flag_1 = (distance_size>0)?(((x_diff*x_diff)+(y_diff*y_diff))<=(distance_size*distance_size)):true;
				if (flag_0&&flag_1) {

						/* Determine the mse */
						cv::Mat q_patch_valu = output_array_0(q_patch_rect_mask);
						mse_patch_ptr[col] = cv::sum(cv::abs(q_patch_valu-p_patch_valu).mul(p_patch_mask))[0];
				}
			}
		}
				
		/* Determine the replacement location which is defined by the minimum mse in the area that is
			not the occlusion */
		cv::Point r_patch_loc;
		cv::minMaxLoc(mse_patch,NULL,NULL,&r_patch_loc,NULL,(mask_array_1==0)&border_array);

		/* Determine the replacement patch's values and mask */
		cv::Rect r_patch_rect_mask(r_patch_loc-patch_offset,patch_size);
		cv::Mat r_patch_valu = output_array_0(r_patch_rect_mask);
		cv::Mat r_patch_mask = (mask_array_1(p_patch_rect_mask)!=0)&1;

		/* Also, let's set the mask values to 0 to indicate the pixels
			have been filled and set the confidence pixel values to that
			of the patch */
		cv::Mat m_patch_valu = mask_array_1(p_patch_rect_mask);
		cv::Mat c_patch_valu = confidence_pixel(p_patch_rect_mask);
		float p_patch_confid = confidence_patch.at<float>(p_patch_loc);

		/* Perform replacement */
		for (int row=0;row<p_patch_valu.rows;row++) {
			logical_type* r_patch_mask_ptr = r_patch_mask.ptr<logical_type>(row);
			logical_type* m_patch_valu_ptr = m_patch_valu.ptr<logical_type>(row);
			float* p_patch_valu_ptr = p_patch_valu.ptr<float>(row);
			float* r_patch_valu_ptr = r_patch_valu.ptr<float>(row);
			float* c_patch_valu_ptr = c_patch_valu.ptr<float>(row);
			for (int col=0;col<p_patch_valu.cols;col++) {
				if (r_patch_mask_ptr[col]!=0) {
					p_patch_valu_ptr[col] = r_patch_valu_ptr[col];
					m_patch_valu_ptr[col] = 0;
					c_patch_valu_ptr[col] = p_patch_confid;
				}
			}
		}

		/* Increment the iterations counter */
		iters++;
	}
};