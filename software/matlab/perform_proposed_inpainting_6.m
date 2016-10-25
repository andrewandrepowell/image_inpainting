function image_array_0 = perform_proposed_inpainting_6(image_array_0,mask_array_0,...
    patch_size,distance_size,skip_factor_row,skip_factor_col,mse_thress,epsilons)

%% Setup - Exem 

image_size = size(image_array_0);
[rows,cols] = meshgrid(...
    1:skip_factor_row:image_size(1),...
    1:skip_factor_col:image_size(2));
indices_0 = 1:numel(image_array_0);
indices_1 = sort(sub2ind(image_size,rows(:),cols(:)))';
skip_indices = 1:numel(indices_1);

distance_thres = distance_size^2;
mask_array_1 = mask_array_0;
alpha = 1;

patch_area = patch_size^2;
patch_mask = ones(patch_size);

dilate_mask_size = 3;
dilate_mask = strel('square',dilate_mask_size);

fill_front_size = 3;
fill_front_area = fill_front_size^2;
fill_front_mask = ones(fill_front_size);

grad_mask_row = fspecial('sobel');
grad_mask_col = grad_mask_row.';

confidence_pixel = zeros(image_size);
confidence_pixel(mask_array_0==false) = 1;
confidence_patch = zeros(image_size);

% Testing software
% handle_mse_array = figure;
handle_image_array = figure;

%% Main Loop

while any(mask_array_0(:))
    
    % Determine the fill front
    fill_front = (imfilter(mask_array_0,fill_front_mask)<...
        fill_front_area)&mask_array_0;
    
    % Determine n (normal to the occlusion
    n_row = imfilter(mask_array_0,grad_mask_row);
    n_col = imfilter(mask_array_0,grad_mask_col);
    
    % Determine patch n
    patch_n_row = n_row;
    patch_n_row(mask_array_0==false) = 0;
    patch_n_col = n_col;
    patch_n_col(mask_array_0==false) = 0;
    
    % Determine isophotes
    iso_row = -imfilter(image_array_0,grad_mask_col);
    iso_col = imfilter(image_array_0,grad_mask_row);
    
    % Determine patch isophotes and confidences
    patch_iso_row = zeros(image_size);
    patch_iso_col = zeros(image_size);
    parfor index = indices_0
        
        % This operation is only done on the fill front
        if fill_front(index) == true
            
            [row,col] = ind2sub(image_size,index);
            location = [row col];
            
            % Determine the confidences of each patch
            confidence_patch_0 = get_patch(confidence_pixel,location,patch_mask);
            confidence_mask_0 = get_patch(~mask_array_0,location,patch_mask);
            confidence_patch(index) = sum(confidence_patch_0(:).*...
                confidence_mask_0(:))/patch_area;
            
            % Get the corresponding patch
            iso_row_0 = get_patch(iso_row,location,patch_mask);
            iso_col_0 = get_patch(iso_col,location,patch_mask);
            iso_mask_0 = imerode(confidence_mask_0,dilate_mask);
            
            % The average isophote of the patch, excluding those in the
            % occlusion, is the isophote of the patch
            patch_iso_total_0 = sum(iso_mask_0(:));
            if patch_iso_total_0 ~=0
                patch_iso_row(index) = sum(iso_row_0(:).*iso_mask_0(:))/...
                    patch_iso_total_0;
                patch_iso_col(index) = sum(iso_col_0(:).*iso_mask_0(:))/...
                    patch_iso_total_0;
            end  
        end
    end
    
    % Calculate data and priority terms, and then find the patch with the
    % largest priority
    data_patch = abs(patch_iso_row.*patch_n_col+patch_iso_col.*patch_n_col)/alpha;
    priority_patch = data_patch.*confidence_patch;
    [priority_max_value,priority_max_index] = max(priority_patch(:));
    if priority_max_value==0
        priority_max_index = find(mask_array_0(:),1,'first');
    end
    [patch_p_row,patch_p_col] = ind2sub(image_size,priority_max_index);
    
    % Determine important matrices concerning the replacement patch p
    patch_p_location = [patch_p_row patch_p_col];
    patch_p_valu = get_patch(image_array_0,patch_p_location,patch_mask);
    patch_p_mask = get_patch(~mask_array_0,patch_p_location,patch_mask);
    
    % Find the most similar patch q
    mse_array = repmat(struct('mse',inf,'loc',[]),...
        size(skip_indices));
    parfor skip_index=skip_indices
        
        % Get true index
        index = indices_1(skip_index);
        
        % Initialize possible mse
        mse_mse = inf;
        mse_loc = [];
        
        % Compute square distance from p's location
        [row,col] = ind2sub(image_size,index);
        patch_q_location = [row col];
        patch_q_distance_fromp = sum((patch_p_location-patch_q_location).^2);
        
        % Only perform operation on patches outside of the occlusion and
        % within specified distance range
        if (distance_thres>patch_q_distance_fromp) && (mask_array_0(index) == false)
            
            % Determine important matrices concerning the similar patch q
            patch_q_location = [row col];
            patch_q_valu = get_patch(image_array_0,patch_q_location,patch_mask);
            patch_q_mask = get_patch(~mask_array_0,patch_q_location,patch_mask);
            
            % Only begin to check if the p-mask is a subject of the q-mask
            if (patch_p_mask&patch_q_mask)==patch_p_mask
                patch_mse = sum(abs((patch_p_valu(:)-patch_q_valu(:)).*...
                    patch_p_mask(:)));
                mse_mse = patch_mse;
                mse_loc = patch_q_location;
            end
        end
        
        % Store data
        mse_array(skip_index).mse = mse_mse;
        mse_array(skip_index).loc = mse_loc;
    end
    
    % Get the mse replacement patch 
    [~,mse_index] = min([mse_array.mse]);
    mse_location = mse_array(mse_index).loc;
    
    % Get the replacement patch q
    patch_q_valu = get_patch(image_array_0,mse_location,patch_mask);
    patch_q_mask = get_patch(~mask_array_0,mse_location,patch_mask);
    
    % Update image, mask, and confidence's of pixels
    patch_replacement_mask = ~patch_p_mask&patch_q_mask;
    image_array_0 = set_patch(image_array_0,patch_q_valu,patch_p_location,patch_replacement_mask);
    mask_array_0 = set_patch(mask_array_0,zeros(patch_size),patch_p_location,patch_replacement_mask);
    confidence_pixel = set_patch(confidence_pixel,...
        confidence_patch(patch_p_row,patch_p_col)*ones(patch_size),...
        patch_p_location,patch_replacement_mask);
    
    % Testing software
%     patch_iso_abs = sqrt(patch_iso_row.^2+patch_iso_col.^2);
%     figure; imshow(patch_iso_abs,[]); title('patch\_iso\_abs');
%     figure; imshow(confidence_patch,[]); title('confidence_patch');
%     figure; quiver(cols,fliplr(rows),patch_n_col,patch_n_row); title('patch\_n');
%     figure; imshow(data_patch,[]); title('data\_patch');
%     figure; imshow(priority_patch,[]); title('priority\_patch');
%    close(handle_mse_array); figure(handle_mse_array); imshow(mse_array,[]); title('mse\_array');
    close(handle_image_array);figure(handle_image_array);imshow(image_array_0);title('image\_array');
    disp(['Remaining pixels: ' num2str(sum(mask_array_0(:)))]);
end

% End with a few rounds of Gillette's variation of Cahn-Hilliard
image_array_0 = perform_cahn_hilliard_gillette_inpainting_2(...
    image_array_0,mask_array_1,mse_thress,epsilons);

end

function patch = get_patch(array,location,mask)

array_size = size(array);
mask_size = numel(mask(:,1));  % assumes square
mask_midp = ceil(mask_size./2);      
select_mask = zeros(2,mask_size);
initial_mask = select_mask;

for index = 1:2
    initial_mask(index,:) = [1:mask_size]+location(index)-mask_midp;
    select_mask(index,:) = max(initial_mask(index,:),1);
    select_mask(index,:) = min(select_mask(index,:),array_size(index));
end

patch = array(select_mask(1,:),select_mask(2,:)).*mask;

for index = 1:2
    correct_mask = (initial_mask(index,:)~=select_mask(index,:));
    patch(correct_mask,:) = 0;
    patch = patch.';
end
    
end

function array = set_patch(array,patch,location,mask)

array_size = size(array);
mask_size = numel(mask(:,1));  % assumes square
mask_midp = ceil(mask_size./2);      
select_mask = zeros(2,mask_size);
initial_mask = select_mask;
patch_mask = select_mask;

for index = 1:2
    initial_mask(index,:) = [1:mask_size]+location(index)-mask_midp;
    select_mask(index,:) = max(initial_mask(index,:),1);
    select_mask(index,:) = min(select_mask(index,:),array_size(index));
    patch_mask(index,:) = select_mask(index,:)-location(index)+mask_midp;
end

patch = patch(patch_mask(1,:),patch_mask(2,:));
mask = mask(patch_mask(1,:),patch_mask(2,:));

original_patch = array(select_mask(1,:),select_mask(2,:));
array(select_mask(1,:),select_mask(2,:)) = patch.*mask + ...
    original_patch.*(~mask);

end

