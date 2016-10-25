function image_array_0 = perform_cahn_hilliard_gillette_inpainting_2(...
    image_array_0,mask_array,mse_thress,epsilons)

lambda_0 = 1;
c2 = lambda_0*1000;
delta_t = 1;
image_size = size(image_array_0);

lambda_array = zeros(image_size);
lambda_array(mask_array==false) = lambda_0;

M = zeros(image_size);
get_exp=@(index,nindices)exp(2*pi*1j*index/nindices);
for row=1:image_size(1)
for col=1:image_size(2)
    M(row,col)=get_exp(-row,image_size(1))+get_exp(row,image_size(1))+...
        get_exp(-col,image_size(2))+get_exp(col,image_size(2))-4;
end
end

image_array_1 = image_array_0;
mse_norm_factor = sum(mask_array(:))^2/1e10;
original_array = image_array_0;
iter = 0;

for stage=1:numel(mse_thress)
    
    mse_thres = mse_thress(stage);
    epsilon = epsilons(stage);
    c1 = 1/epsilon;
    Denom = 1+epsilon*M.^2-c1*M+c2;
    mse_0 = inf;
    
    while mse_thres<mse_0

        Image_array = fft2(image_array_0);
        W_deriv = fft2(4*image_array_0.^3-6*image_array_0.^2+2*image_array_0);
        Term_0 = delta_t*M.*(W_deriv/epsilon-c1*Image_array);
        Term_1 = delta_t*fft2(lambda_array.*original_array+(c2-lambda_array)...
            .*image_array_0);
        Numer = (Term_0+Term_1+Image_array);
        image_array_0 = real(ifft2(Numer./Denom));
        image_array_0 = max(image_array_0,0);
        image_array_0 = min(image_array_0,1);
        
        select_mask = mask_array==true;
        mse_0 = mean((image_array_1(select_mask)-image_array_0(select_mask)).^2)/...
            mse_norm_factor;
        image_array_1 = image_array_0;
        
        % debug code - comment this out when not using
        iter = iter+1;
        if mod(iter,500)==0
            iter
            mse_0
            close(gcf); figure; imshow(image_array_0);
        end
    end
end

end