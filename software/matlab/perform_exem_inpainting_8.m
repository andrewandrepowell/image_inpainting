function [output_array,iters] = perform_exem_inpainting_8(input_array,mask_array,...
    patch_size)

[output_array,iters] = perform_exem_inpainting_mex_0(...
        single(input_array),single(mask_array),int32(patch_size));

end


