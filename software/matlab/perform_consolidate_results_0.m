% Clear memory
close all;
clear all;

% Important constants
save_dir = 'C:\Users\Andrew Powell\Documents\Current Courses\ECE 9524 Digital Image Processing\final_project\final_report\';

% Load the results
result_name = 'perform_evaluation_0_results.mat';
load(result_name)
result_structs = cell2mat(result_structs);

for i=1:numel(result_structs)
    
    % Extract the data
    result_struct = result_structs(i);
    image_string = result_struct.image_string;
    algorithm_string = result_struct.algorithm_string;
    trial = num2str(result_struct.trial);
    iters = num2str(result_struct.iters);
    inpainted_array = result_struct.inpainted_array;
    time_elapsed = num2str(result_struct.time_elapsed);
    
%     title_string = char(...
%         ['Algorithm: ' algorithm_string ' , Trial: ' trial ', Time: ' time_elapsed ' s']);
    title_string = char(...
        ['Algorithm: ' algorithm_string ', Time: ' time_elapsed ' s']);
    save_string = [save_dir 'imag_' image_string '_alg_' algorithm_string '_tri_' trial '.png'];
    
    display_string = [save_string(1:end-4) ', Iters: ' iters];
    disp(display_string);
    
    % Display figure
    figure;
    imshow(inpainted_array);
    title(title_string);
    reduceWhiteSpace;
    hgexport(gcf,save_string,hgexport('factorystyle'),'Format','png');
    
    
end