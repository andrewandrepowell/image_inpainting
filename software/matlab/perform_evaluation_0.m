% This script is the final script for evaluating the selected approaches to
% inpainting, which includes the proposed approach which in reality is
% a combination, with a few other modifications, of the alternative
% approaches to inpainting

% First, ensure the memory is cleared and there are no windows already open
close all;
clear all;

%% Images

load_image = @(image_name)im2double(rgb2gray(imread(image_name)));
create_image_struct = @(image_array,mask_array,string)hstruct(struct(...
    'image_array',image_array,...
    'mask_array',mask_array,...
    'string',string));
image_structs = {};

% Simplistic Mask
image_size = [50 100];
rows = 1:image_size(1);
cols = 1:image_size(2);
mask_array = zeros(image_size);
mask_array(rows>=10&rows<=40,cols>=45&cols<=55) = true;
select_vec = mask_array==true;
create_random_array = @(image_array)rand([numel(image_array) 1]);

% Simplistic Bar
image_name = 'simple_image_0.png';
image_array = load_image(image_name);
image_array(select_vec) = create_random_array(image_array(select_vec));
image_bar_struct = create_image_struct(image_array,mask_array,'bar');
image_structs{end+1} = image_bar_struct;

% Simplistic Ball
image_name = 'simple_image_1.png';
image_array = load_image(image_name);
image_array = imfilter(image_array,fspecial('average',3));
image_array(select_vec) = create_random_array(image_array(select_vec));
image_ball_struct = create_image_struct(image_array,mask_array,'ball');
image_structs{end+1} = image_ball_struct;

% Simplistic Slide
image_name = 'simple_image_2.png';
image_array = load_image(image_name);
image_array = imfilter(image_array,fspecial('average',5));
image_array(select_vec) = create_random_array(image_array(select_vec));
image_slide_struct = create_image_struct(image_array,mask_array,'slide');
image_structs{end+1} = image_slide_struct;

% Simplistic Balls
image_name = 'simple_image_3.png';
image_array = load_image(image_name);
image_array = imfilter(image_array,fspecial('average',7));
image_array(select_vec) = create_random_array(image_array(select_vec));
image_balls_struct = create_image_struct(image_array,mask_array,'balls');
image_structs{end+1} = image_balls_struct;

% Complex Mask
mask_color = [163,73,164];
create_mask_array = @(color_array)...
    (color_array(:,:,1)==mask_color(1))&...
    (color_array(:,:,2)==mask_color(2))&...
    (color_array(:,:,3)==mask_color(3));

% Complex Einstein
image_name = 'complex_image_0_mask.png';
image_array = load_image(image_name);
image_array = imfilter(image_array,fspecial('average',3));
color_array = imread(image_name);
mask_array = create_mask_array(color_array);
select_vec = mask_array==true;
image_array(select_vec) = create_random_array(image_array(select_vec));
image_einstein_struct = create_image_struct(image_array,mask_array,'einstein');
image_structs{end+1} = image_einstein_struct;

% Complex Waterfall
image_name = 'complex_image_1_mask.png';
image_array = load_image(image_name);
color_array = imread(image_name);
mask_array = create_mask_array(color_array);
select_vec = mask_array==true;
image_array(select_vec) = create_random_array(image_array(select_vec));
image_waterfall_struct = create_image_struct(image_array,mask_array,'waterfall');
image_structs{end+1} = image_waterfall_struct;

% % Display Images
% for image_struct=image_structs
%     figure;
%     imshow(image_struct{1}.d.image_array);
%     title(image_struct{1}.d.string);
%     figure;
%     imshow(image_struct{1}.d.mask_array);
%     title(['Mask: ' image_struct{1}.d.string]);
% end

%% Algorithms

create_algorithm_struct = @(func,param_structs,string)hstruct(struct(...
    'func',func,...
    'param_structs',{param_structs},...
    'string',string));
create_param_struct = @(image_string,params)hstruct(struct(...
    'image_string',image_string,'params',{params}));
algorithm_structs = {};

% Cahn-Hilliar - parameters - default
param_structs = {};
mse_thress = [1e-7,1e-10];
epsilons = [7,1];
params = {mse_thress,epsilons};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Cahn-Hilliard - struct
func = @perform_cahn_hilliard_gillette_inpainting_2;
algorithm_cahn_struct = create_algorithm_struct(func,param_structs,'cahn');
algorithm_structs{end+1} = algorithm_cahn_struct;

% Bertalmio - parameters - default
param_structs = {};
total_inpaint_iters = 2;
total_anidiffuse_iters = 6;
mse_thress = [1,1e-5];
delta_ts = [0.05,0.001];
sensitivities = [100,1];
params = {total_inpaint_iters,total_anidiffuse_iters,mse_thress,delta_ts,sensitivities};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Bertalmio - parameters - ball
total_inpaint_iters = 2;
total_anidiffuse_iters = 6;
mse_thress = [1,1e-5];
delta_ts = [0.05,0.001];
sensitivities = [100,100];
params = {total_inpaint_iters,total_anidiffuse_iters,mse_thress,delta_ts,sensitivities};
ball_param_struct = create_param_struct('ball',params);
param_structs{end+1} = ball_param_struct;

% Bertalmio - parameters - slide
total_inpaint_iters = 2;
total_anidiffuse_iters = 6;
mse_thress = [1e-5,1e-5];
delta_ts = [0.05,0.002];
sensitivities = [100,1];
params = {total_inpaint_iters,total_anidiffuse_iters,mse_thress,delta_ts,sensitivities};
slide_param_struct = create_param_struct('slide',params);
param_structs{end+1} = slide_param_struct;

% Bertalmio - struct
func = @perform_bertalmio_pde_inpainting_2;
algorithm_bertalmio_struct = create_algorithm_struct(func,param_structs,'bertalmio');
algorithm_structs{end+1} = algorithm_bertalmio_struct;

% Exem - parameters - default
param_structs = {};
patch_size = 9;
distance_size = inf;    % In the original exem algorithm, distance is not considered
skip_factor = 1;        % In the original exem algorithm, patches aren't skipped
params = {patch_size,distance_size,skip_factor};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Exem - parameters - waterfall
patch_size = 15;
distance_size = inf;    % In the original exem algorithm, distance is not considered
skip_factor = 1;        % In the original exem algorithm, patches aren't skipped
params = {patch_size,distance_size,skip_factor};
waterfall_param_struct = create_param_struct('waterfall',params);
param_structs{end+1} = waterfall_param_struct;

% Exem - struct
func = @perform_exem_inpainting_7;
algorithm_exem_struct = create_algorithm_struct(func,param_structs,'exem');
algorithm_structs{end+1} = algorithm_exem_struct;

% Proposed - parameters - default
param_structs = {};
patch_size = 7;
distance_size = patch_size*10; 
skip_factor_row = 2;
skip_factor_col = 2;
mse_thress = 1e-5;
epsilons = 1;
params = {patch_size,distance_size,skip_factor_row,skip_factor_col,...
    mse_thress,epsilons};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Proposed - parameters - bar
patch_size = 7;
distance_size = patch_size*10; 
skip_factor_row = 1;
skip_factor_col = 10;
mse_thress = 1;
epsilons = 1;
params = {patch_size,distance_size,skip_factor_row,skip_factor_col,...
    mse_thress,epsilons};
bar_param_struct = create_param_struct('bar',params);
param_structs{end+1} = bar_param_struct;

% Proposed - parameters - ball
patch_size = 7;
distance_size = patch_size*100; 
skip_factor_row = 1;
skip_factor_col = 10;
mse_thress = [1e-7,1e-10];
epsilons = [7,1];
params = {patch_size,distance_size,skip_factor_row,skip_factor_col,...
    mse_thress,epsilons};
ball_param_struct = create_param_struct('ball',params);
param_structs{end+1} = ball_param_struct;

% Proposed - parameters - einstein
patch_size = 7;
distance_size = patch_size*4; 
skip_factor_row = 1;
skip_factor_col = 1;
mse_thress = [1e-5];
epsilons = [1];
params = {patch_size,distance_size,skip_factor_row,skip_factor_col,...
    mse_thress,epsilons};
einstein_param_struct = create_param_struct('einstein',params);
param_structs{end+1} = einstein_param_struct;

% Proposed - parameters - waterfall
patch_size = 40;
distance_size = patch_size; 
skip_factor_row = 1;
skip_factor_col = patch_size;
mse_thress = [1e-5];
epsilons = [1];
params = {patch_size,distance_size,skip_factor_row,skip_factor_col,...
    mse_thress,epsilons};
waterfall_param_struct = create_param_struct('waterfall',params);
param_structs{end+1} = waterfall_param_struct;

% Proposed - struct
func = @perform_proposed_inpainting_6;
algorithm_proposed_struct = create_algorithm_struct(func,param_structs,'proposed');
algorithm_structs{end+1} = algorithm_proposed_struct;


%% Perform 

% Instead of storing by reference, we want to store with the raw data
create_result_struct = @(image_string,algorithm_string,trial,...
    inpainted_array,time_elapsed)...
    struct(...
    'image_string',image_string,...
    'algorithm_string',algorithm_string,...
    'trial',trial,...
    'inpainted_array',inpainted_array,...
    'time_elapsed',time_elapsed);
ntrials = 1;

% Perform cartesian product between image structs, algorithm structs, and
% trials
iters_image_structs = 1:numel(image_structs);
iters_algorithm_structs = 1:numel(algorithm_structs);
iters_ntrials = 1:ntrials;
[mesh_images,mesh_algorithms,mesh_trials] = ...
    meshgrid(iters_image_structs,iters_algorithm_structs,iters_ntrials);
performances = [mesh_images(:),mesh_algorithms(:),mesh_trials(:)];
iters = 1:numel(performances(:,1));

% Load any results that have already been saved, but create a new cell
% array if no results have been created
results_name = 'perform_evaluation_0_results.mat';
try
    load(results_name);
catch
    result_structs = cell(iters(end),1);
end

% Perform evaluation
for iter=iters
    
    % Grab data/handles
    image_struct = image_structs{performances(iter,1)};
    algorithm_struct = algorithm_structs{performances(iter,2)};
    trial = performances(iter,3);
    
    % Report information
    disp('Reporting information...');
    disp(char(...
        ['Image: ' image_struct.d.string],...
        ['Algorithm: ' algorithm_struct.d.string]));
    
    % Only perform evaluation if results don't already exist
    if ~isempty(result_structs{iter}), disp('skipped...'); continue; end;
    
    % Acquire parameters
    image_string = image_struct.d.string;
    param_structs = algorithm_struct.d.param_structs;
    for param_struct=param_structs
        param_image_string = param_struct{1}.d.image_string;
        if strcmp('default',param_image_string)|| ...
                strcmp(image_string,param_image_string)
            algorithm_param_struct = param_struct{1};
        end
    end
    disp(['Params: ' param_image_string]);
    
    % Run algorithm
    disp('Running algorithm...');
    tic;
    inpainted_array = algorithm_struct.d.func(image_struct.d.image_array,...
        image_struct.d.mask_array,algorithm_param_struct.d.params{:});
    time_elapsed = toc;
    
    % Display inpainted image
    figure;
    imshow(inpainted_array);
    title(['Image: ' image_struct.d.string ...
        ', Algorithm: ' algorithm_struct.d.string]);
    
    % Report final information
    disp('Algorithm finished...');
    disp(char(...
        ['Time Elapsed: ' num2str(time_elapsed)]));
    
    % Store results
    image_string = image_struct.d.string;
    algorithm_string = algorithm_struct.d.string;
    result_structs{iter} = create_result_struct(...
        image_string,algorithm_string,trial,inpainted_array,time_elapsed);
    save(results_name,'result_structs'); % In case of crash, ALWAYS save
end


