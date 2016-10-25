% This script is the final script for evaluating the selected approaches to
% inpainting, which includes the proposed approach which in reality is
% a combination, with a few other modifications, of the alternative
% approaches to inpainting

% First, ensure the memory is cleared and there are no windows already open
close all;
clear all;

%% Images

save_dir = 'C:\Users\Andrew Powell\Documents\Current Courses\ECE 9524 Digital Image Processing\final_project\final_report\';
load_image = @(image_name)im2single(rgb2gray(imread(image_name)));
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
mask_dilate = strel('square',3);
mask_color = [163,73,164];
create_mask_array = @(color_array)imdilate(...
    (color_array(:,:,1)==mask_color(1))&...
    (color_array(:,:,2)==mask_color(2))&...
    (color_array(:,:,3)==mask_color(3)),...
    mask_dilate);

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
%     reduceWhiteSpace;
%     
%     save_string = [save_dir 'img_' image_struct{1}.d.string '.png'];
%     hgexport(gcf,save_string,hgexport('factorystyle'),'Format','png');
%     
%     figure;
%     imshow(image_struct{1}.d.mask_array);
%     title(['Mask: ' image_struct{1}.d.string]);
%     reduceWhiteSpace;
%     
%     save_string = [save_dir 'img_' image_struct{1}.d.string '_mas.png'];
%     hgexport(gcf,save_string,hgexport('factorystyle'),'Format','png');
%     
%     display_string = char(...
%         ['Image: ' image_struct{1}.d.string],...
%         ['Mask Area: ' num2str(sum(image_struct{1}.d.mask_array(:)))]);
%     disp(display_string);
% end

%% Algorithms

create_algorithm_struct = @(func,param_structs,string,iters_flag)hstruct(struct(...
    'func',func,...
    'param_structs',{param_structs},...
    'string',string,...
    'iters_flag',iters_flag));
create_param_struct = @(image_string,params)hstruct(struct(...
    'image_string',image_string,'params',{params}));
algorithm_structs = {};

% Cahn-Hilliar - parameters - default
param_structs = {};
total_iters = [10000,15000];
epsilons = [5,.5];
total_stages = 2;
params = {total_iters,epsilons,total_stages};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Cahn-Hilliar - parameters - slide
total_iters = [10000,10000];
epsilons = [5,3];
total_stages = 2;
params = {total_iters,epsilons,total_stages};
default_param_struct = create_param_struct('slide',params);
param_structs{end+1} = default_param_struct;

% Cahn-Hilliar - parameters - balls
total_iters = [10000,10000];
epsilons = [3,2];
total_stages = 2;
params = {total_iters,epsilons,total_stages};
default_param_struct = create_param_struct('balls',params);
param_structs{end+1} = default_param_struct;

% Cahn-Hilliar - parameters - einstein
total_iters = [10000,100];
epsilons = [2,1];
total_stages = 2;
params = {total_iters,epsilons,total_stages};
einstein_param_struct = create_param_struct('einstein',params);
param_structs{end+1} = einstein_param_struct;

% Cahn-Hilliar - parameters - waterfall
total_iters = [100];
epsilons = [3];
total_stages = 1;
params = {total_iters,epsilons,total_stages};
waterfall_param_struct = create_param_struct('waterfall',params);
param_structs{end+1} = waterfall_param_struct;

% Cahn-Hilliard - struct
func = @perform_cahn_hilliard_gillette_inpainting_3;
algorithm_cahn_struct = create_algorithm_struct(func,param_structs,'cahn',false);
algorithm_structs{end+1} = algorithm_cahn_struct;

% Bertalmio - parameters - default
param_structs = {};
total_iters = [10000,20000];
total_inpaint_iters = [2,2];
total_anidiffuse_iters = [2,2];
total_stages = 2;
delta_ts = [0.001,0.00001];
sensitivities = [100,.1];
diffuse_coef = [1,1];
params = {total_iters,total_inpaint_iters,total_anidiffuse_iters,total_stages,...
    delta_ts,sensitivities,diffuse_coef};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Bertalmio - parameters - slide
total_iters = [10000,10000];
total_inpaint_iters = [2,2];
total_anidiffuse_iters = [2,2];
total_stages = 2;
delta_ts = [0.001,0.001];
sensitivities = [100,.1];
diffuse_coef = [1,1];
params = {total_iters,total_inpaint_iters,total_anidiffuse_iters,total_stages,...
    delta_ts,sensitivities,diffuse_coef};
slide_param_struct = create_param_struct('slide',params);
param_structs{end+1} = slide_param_struct;

% Bertalmio - struct
func = @perform_bertalmio_pde_inpainting_3;
algorithm_bertalmio_struct = create_algorithm_struct(func,param_structs,'bertalmio',false);
algorithm_structs{end+1} = algorithm_bertalmio_struct;

% Exem - parameters - default
param_structs = {};
patch_size = 20;
params = {patch_size};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;
 
% Exem - parameters - balls
patch_size = 7;
params = {patch_size};
balls_param_struct = create_param_struct('balls',params);
param_structs{end+1} = balls_param_struct;

% Exem - parameters - einstein
patch_size = 9;
params = {patch_size};
einstein_param_struct = create_param_struct('einstein',params);
param_structs{end+1} = einstein_param_struct;

% Exem - parameters - waterfall
patch_size = 70;
params = {patch_size};
waterfall_param_struct = create_param_struct('waterfall',params);
param_structs{end+1} = waterfall_param_struct;

% Exem - struct
func = @perform_exem_inpainting_8;
algorithm_exem_struct = create_algorithm_struct(func,param_structs,'exem',true);
algorithm_structs{end+1} = algorithm_exem_struct;

% Proposed - parameters - default
param_structs = {};
patch_size = 15;
distance_size = patch_size*10; 
skip_factor = [1 1];
cahn_epsilons = 1;
cahn_total_iters = 100;
total_stages = 1;
display_name = 'Proposed Window';
params = {patch_size,distance_size,skip_factor,cahn_epsilons,...
    cahn_total_iters,total_stages};
default_param_struct = create_param_struct('default',params);
param_structs{end+1} = default_param_struct;

% Proposed - parameters - bar
patch_size = 15;
distance_size = patch_size*10; 
skip_factor = [2 1];
cahn_epsilons = [];
cahn_total_iters = [];
total_stages = 0;
display_name = 'Proposed Window';
params = {patch_size,distance_size,skip_factor,cahn_epsilons,...
    cahn_total_iters,total_stages};
bar_param_struct = create_param_struct('bar',params);
param_structs{end+1} = bar_param_struct;

% Proposed - parameters - ball
patch_size = 15;
distance_size = patch_size*10; 
skip_factor = [1 1];
cahn_epsilons = [5,.5];
cahn_total_iters = [10000,10000];
total_stages = 2;
display_name = 'Proposed Window';
params = {patch_size,distance_size,skip_factor,cahn_epsilons,...
    cahn_total_iters,total_stages};
ball_param_struct = create_param_struct('ball',params);
param_structs{end+1} = ball_param_struct;

% Proposed - parameters - balls
patch_size = 7;
distance_size = 0; 
skip_factor = [1 1];
cahn_epsilons = 1;
cahn_total_iters = 0;
total_stages = 1;
display_name = 'Proposed Window';
params = {patch_size,distance_size,skip_factor,cahn_epsilons,...
    cahn_total_iters,total_stages,display_name};
balls_param_struct = create_param_struct('balls',params);
param_structs{end+1} = balls_param_struct;

% Proposed - parameters - einstein
patch_size = 7;
distance_size = patch_size*4; 
skip_factor = [1 1];
cahn_epsilons = 2;
cahn_total_iters = 100;
total_stages = 1;
display_name = 'Proposed Window';
params = {patch_size,distance_size,skip_factor,cahn_epsilons,...
    cahn_total_iters,total_stages};
einstein_param_struct = create_param_struct('einstein',params);
param_structs{end+1} = einstein_param_struct;

% Proposed - parameters - waterfall
patch_size = 70;
distance_size = 0; 
skip_factor = [2 1];
cahn_epsilons = 2;
cahn_total_iters = 1;
total_stages = 1;
display_name = 'Proposed Window';
params = {patch_size,distance_size,skip_factor,cahn_epsilons,...
    cahn_total_iters,total_stages};
waterfall_param_struct = create_param_struct('waterfall',params);
param_structs{end+1} = waterfall_param_struct;

% Proposed - struct
func = @perform_proposed_inpainting_7;
algorithm_proposed_struct = create_algorithm_struct(func,param_structs,'proposed',true);
algorithm_structs{end+1} = algorithm_proposed_struct;


%% Perform 

% Instead of storing by reference, we want to store with the raw data
create_result_struct = @(image_string,algorithm_string,trial,...
    iters,mask_array,inpainted_array,time_elapsed)...
    struct(...
    'image_string',image_string,...
    'algorithm_string',algorithm_string,...
    'trial',trial,...,
    'iters',iters,...
    'mask_array',mask_array,...
    'inpainted_array',inpainted_array,...
    'time_elapsed',time_elapsed);
ntrials = 3;

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
    disp(['Params: ' algorithm_param_struct.d.image_string]);
    
    % Run algorithm
    disp('Running algorithm...');
    iters = -1;
    if (algorithm_struct.d.iters_flag==false)
        tic;
        [inpainted_array] = algorithm_struct.d.func(image_struct.d.image_array,...
            image_struct.d.mask_array,algorithm_param_struct.d.params{:});
        time_elapsed = toc;
    else
        tic;
        [inpainted_array,iters] = algorithm_struct.d.func(image_struct.d.image_array,...
            image_struct.d.mask_array,algorithm_param_struct.d.params{:});
        time_elapsed = toc;
    end
    
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
        image_string,algorithm_string,trial,iters,mask_array,inpainted_array,time_elapsed);
    save(results_name,'result_structs'); % In case of crash, ALWAYS save
end


