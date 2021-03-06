#include "mex.h"
#include "./visual_studios/inpainting_algorithm/shared_library/shared_library.h"
 
/* 
%Release compile command
OCVRoot = 'C:\Users\Andrew Powell\Documents\opencv\build';
IPath_0 = ['-I',fullfile(OCVRoot,'include')];
IPath_1 = ['-I','.\visual_studios\inpainting_algorithm\shared_library'];
LPath_0 = fullfile(OCVRoot,'x64\vc11\staticlib');
LPath_1 = ['.\visual_studios\inpainting_algorithm\x64\Release'];
libs = {...
    'kernel32.lib',...
    'user32.lib',...
    'gdi32.lib',...
    'winspool.lib',...
    'shell32.lib',...
    'ole32.lib',...
    'oleaut32.lib',...
    'uuid.lib',...
    'comdlg32.lib',...
    'advapi32.lib',...
    'comctl32.lib',...
    'setupapi.lib',...
    'ws2_32.lib',...
    'vfw32.lib',...
	fullfile(LPath_1,'shared_library.lib'),...
    fullfile(LPath_0,'IlmImf.lib'),...
    fullfile(LPath_0,'libjasper.lib'),...
    fullfile(LPath_0,'libjpeg.lib'),... 
    fullfile(LPath_0,'libpng.lib'),... 
    fullfile(LPath_0,'libtiff.lib'),... 
    fullfile(LPath_0,'opencv_calib3d2410.lib'),... 
    fullfile(LPath_0,'opencv_contrib2410.lib'),... 
    fullfile(LPath_0,'opencv_core2410.lib'),... 
    fullfile(LPath_0,'opencv_features2d2410.lib'),... 
    fullfile(LPath_0,'opencv_flann2410.lib'),... 
    fullfile(LPath_0,'opencv_gpu2410.lib'),... 
    fullfile(LPath_0,'opencv_highgui2410.lib'),... 
    fullfile(LPath_0,'opencv_imgproc2410.lib'),... 
    fullfile(LPath_0,'opencv_legacy2410.lib'),... 
    fullfile(LPath_0,'opencv_ml2410.lib'),...
    fullfile(LPath_0,'opencv_nonfree2410.lib'),... 
    fullfile(LPath_0,'opencv_objdetect2410.lib'),... 
    fullfile(LPath_0,'opencv_ocl2410.lib'),...
    fullfile(LPath_0,'opencv_photo2410.lib'),... 
    fullfile(LPath_0,'opencv_stitching2410.lib'),... 
    fullfile(LPath_0,'opencv_superres2410.lib'),... 
    fullfile(LPath_0,'opencv_ts2410.lib'),...
    fullfile(LPath_0,'opencv_video2410.lib'),... 
    fullfile(LPath_0,'opencv_videostab2410.lib'),... 
    fullfile(LPath_0,'zlib.lib')}; 
mex('perform_proposed_inpainting_mex_0.cpp',IPath_0,IPath_1,libs{:},'LINKFLAGS="/NODEFAULTLIB:msvcrt.lib"');
*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    
    /* Declarations */
    int m = mxGetM(prhs[0]); // nrows
    int n = mxGetN(prhs[0]); // ncols
    plhs[0] = mxCreateNumericMatrix(m,n,mxSINGLE_CLASS,mxREAL);  
    plhs[1] = (nlhs>1)?mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL):NULL;
    float* input_array = (float*)mxGetData(prhs[0]);
    float* mask_array = (float*)mxGetData(prhs[1]);
    float* output_array = (float*)mxGetData(plhs[0]);
    int patch_size_0 = mxGetScalar(prhs[2]);
    int distance_size = mxGetScalar(prhs[3]);
    int* skip_factor = (int*)mxGetData(prhs[4]);
    float* cahn_epsilons = (float*)mxGetData(prhs[5]);
    int* cahn_total_iters = (int*)mxGetData(prhs[6]);
    int total_stages = mxGetScalar(prhs[7]);
    char* display_name = (nrhs<=8)?NULL:(char*)mxGetData(prhs[8]);
    int* iters_0 = (nlhs>1)?(int*)mxGetData(plhs[1]):NULL;
    
    /* Run algorithm */
    int iters_1 = perform_proposed_inpainting_0(
        input_array,mask_array,output_array, 
        m,n,
        patch_size_0,distance_size,
        skip_factor,
        cahn_epsilons,cahn_total_iters,total_stages,
        display_name);
    
    if (iters_0!=NULL) {
        *iters_0 = iters_1;
    }
}