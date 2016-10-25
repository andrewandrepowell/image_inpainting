function image_array_0 = perform_bertalmio_pde_inpainting_2(image_array_0,mask_array,...
    total_inpaint_iters,total_anidiffuse_iters,mse_thress,delta_ts,sensitivities)

mask_grad_sobel_row = fspecial('sobel');
mask_grad_sobel_col = mask_grad_sobel_row.';
mask_laplacian = fspecial('laplacian');

diffuse_coef = 1;
if diffuse_coef==0
    compute_diffuse_ceof = @compute_diffuse_coef_0;
else
    compute_diffuse_ceof = @compute_diffuse_coef_1;
end

image_array_1 = image_array_0;
mse_norm_factor = sum(mask_array(:))^2/1e10;

stages = 1:numel(mse_thress);

for stage=stages

    sensitivity = sensitivities(stage);
    delta_t = delta_ts(stage);
    mse_thres = mse_thress(stage);
    mse_0 = inf;
    iter = 0;
    
    while mse_0>mse_thres

        for iter_anidiffuse=1:total_anidiffuse_iters

            image_grad_row = imfilter(image_array_0,mask_grad_sobel_row);
            image_grad_col = imfilter(image_array_0,mask_grad_sobel_col);
            diffuse_coef = compute_diffuse_ceof(image_grad_row,image_grad_col,sensitivity);
            image_laplacian = imfilter(image_array_0,mask_laplacian);

            update_term = delta_t*(diffuse_coef.*image_laplacian);
            select_mask = mask_array==true;
            image_array_0(select_mask) = image_array_0(select_mask)+...
                update_term(select_mask);
        end

        for iter_inpaint=1:total_inpaint_iters

            image_iso_row = -imfilter(image_array_0,mask_grad_sobel_col);
            image_iso_col = imfilter(image_array_0,mask_grad_sobel_row);
            image_iso_norm = sqrt(image_iso_row.^2+image_iso_col.^2);
            image_laplacian = imfilter(image_array_0,mask_laplacian);
            image_laplacian_grad_row = imfilter(image_laplacian,mask_grad_sobel_row);
            image_laplacian_grad_col = imfilter(image_laplacian,mask_grad_sobel_col);

            update_term = delta_t*...
                (image_iso_row.*image_laplacian_grad_row+...
                image_iso_col.*image_laplacian_grad_col)./...
                image_iso_norm;
            select_mask = mask_array==true&~isnan(update_term);
            image_array_0(select_mask) = image_array_0(select_mask)-...
                update_term(select_mask);
            image_array_0 = min(image_array_0,1);
            image_array_0 = max(image_array_0,0);
        end

        select_mask = mask_array==true;
        mse_0 = mean((image_array_1(select_mask)-image_array_0(select_mask)).^2)/...
            mse_norm_factor;
        image_array_1 = image_array_0;

        iter = iter+1;
        if mod(iter,200)==0
            iter
            mse_0
            close(gcf); figure; imshow(image_array_0);
        end

    end

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