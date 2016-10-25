
% close all images
close all;

% load testing image
image_array = im2double(rgb2gray(imread('coin-inpainted.png')));
figure; imshow(image_array);

% perform anisotropic diffusion
total_iters = 100;
diffuse_coef = 2;
sensitivity = 0.11;
delta_t = 0.1;
image_current = perform_perona_malik_ani_diffusion_0(image_array,...
    total_iters,diffuse_coef,sensitivity,delta_t);
figure; imshow(image_current);