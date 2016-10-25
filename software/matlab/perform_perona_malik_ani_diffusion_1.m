function image_array = perform_perona_malik_ani_diffusion_1(image_array,total_iters,diffuse_coef,sensitivity,delta_t)

mask_grad_sobel_row = fspecial('sobel');
mask_grad_sobel_col = mask_grad_sobel_row.';
mask_laplacian = fspecial('laplacian');
if diffuse_coef==0
    compute_diffuse_ceof = @compute_diffuse_coef_0;
else
    compute_diffuse_ceof = @compute_diffuse_coef_1;
end

for iter = 1:total_iters
    
    % Compute necessasry matrices
    image_grad_row = imfilter(image_array,mask_grad_sobel_row);
    image_grad_col = imfilter(image_array,mask_grad_sobel_col);
    diffuse_coef = compute_diffuse_ceof(image_grad_row,image_grad_col,sensitivity);
    image_laplacian = imfilter(image_array,mask_laplacian);
    
    % Update the current image
    image_array = image_array+delta_t*(diffuse_coef.*image_laplacian);
end

end

function diffuse_coef = compute_diffuse_coef_0(image_grad_row,image_grad_col,sensitivity)

image_grad_norm = sqrt(image_grad_row.^2+image_grad_col.^2);
diffuse_coef = exp(-(image_grad_norm/sensitivity));

end

function diffuse_coef = compute_diffuse_coef_1(image_grad_row,image_grad_col,sensitivity)

image_grad_norm = sqrt(image_grad_row.^2+image_grad_col.^2);
diffuse_coef = 1./(1+(image_grad_norm/sensitivity).^2);

end