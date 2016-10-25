#ifndef SHARED_LIBRARY_H_
#define SHARED_LIBRARY_H_



void perform_cahn_hilliard_gillette_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,
	int* total_iters,float* epsilons,int total_stages);

int perform_proposed_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,
	int patch_size_0,int distance_size,
	int* skip_factor,
	float* cahn_epsilons,int* cahn_total_iters,int total_stages,
	char* display_name);

void perform_bertalmio_pde_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,
	int* total_iters,int* total_inpaint_iters, int* total_anidiffuse_iters,int total_stages,
	float* delta_ts,float* sensitivities,int diffuse_coef);

int perform_exem_inpainting_0(
	float* input_array,float* mask_array,float* output_array, 
	int rows, int cols,int patch_size_0);

#endif