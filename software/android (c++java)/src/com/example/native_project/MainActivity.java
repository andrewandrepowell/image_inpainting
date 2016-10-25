package com.example.native_project;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.example.native_project.R;

public class MainActivity extends Activity implements  CvCameraViewListener2, OnTouchListener  {

	private final String TAG = "TAG";
	private final int image_width = 864/2;
	private final int image_height = 480/2;
	private int screen_width;
	private int screen_height;
	private CameraBridgeViewBase mOpenCvCameraView;
	private Mat input_frame = null;
	private Mat mask_frame = new Mat();
	private Mat mask_frame_8u = new Mat();
	private Mat display_frame = new Mat();
	private Mat captured_frame = new Mat();
	private Mat inpainted_frame = new Mat();
	private Mat dilate_frame = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(30,30));
	private boolean screen_captured = false;
	private boolean mask_captured = false;
	private boolean overlap_captured = false;
	private boolean inpainted_captured = false;
	private boolean inpainted_configured = false;
	private long handle = 0;
	private int pixels_total = 0;
	private int pixels_remaining = 0;
	
	static {
	    if (!OpenCVLoader.initDebug()) {
	        // Handle initialization error
	    } else {
	    	// Add native libraries here
	    	System.loadLibrary("native"); 
	    }
	}
	
	private native void native_setup(long input_array_ptr, long mask_array_ptr, long output_array_ptr);
	private native void native_reset();
	private native void native_destroy();
	private native int native_get_pixels();
	private native boolean native_perform();
	private native void native_update();
	private native void perform_proposed_inpainting_0(long input_array_ptr, long mask_array_ptr, long output_array_ptr);
	
    @SuppressLint("ClickableViewAccessibility")
	@Override
    protected void onCreate(Bundle savedInstanceState) {
    	Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setMaxFrameSize(image_width,image_height); 
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(this);
        mOpenCvCameraView.enableView();
        Display display = getWindowManager().getDefaultDisplay();
        screen_width = display.getWidth();
        screen_height = display.getHeight();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
    
    @Override
    public void onResume()
    {
        super.onResume();
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        if ((mOpenCvCameraView!=null)&&
        		(screen_captured==false))
        	mOpenCvCameraView.enableView();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView!=null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView!=null) {
            mOpenCvCameraView.disableView();
        }
        if (handle!=0) {
        	native_destroy();
        }
        mask_frame.release();
        mask_frame_8u.release();
        display_frame.release();
        captured_frame.release();
        inpainted_frame.release();
        dilate_frame.release();
    }

	@Override
	public void onCameraViewStarted(int width, int height) {
	}

	@Override
	public void onCameraViewStopped() {
	}

	@Override
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		
		input_frame = inputFrame.gray();
		
		if ((inpainted_configured==true)&&(inpainted_captured==false)) {
			inpainted_captured = native_perform();
			native_update();
			inpainted_frame.copyTo(display_frame);
			Core.multiply(display_frame, new Scalar(255), display_frame);
			display_frame.convertTo(display_frame, CvType.CV_8U);
		}
		
		if (screen_captured==true||mask_captured==true||
				overlap_captured==true||inpainted_configured==true||
				inpainted_captured==true) {
			return display_frame;
		} else {
			return input_frame;
		}
	}

	@Override
	public boolean onTouch(View v, MotionEvent event) {
	    switch (event.getAction()) {
	    
	    case MotionEvent.ACTION_MOVE:
	    	
	    	// As the user's finger is dragged, the 
	    	// mask is filled
	    	if ((screen_captured==true)&&
	    			(mask_captured==false)) {
	    		int x = (int)event.getX();
	    		int y = (int)event.getY();
	    		double scale = (double)input_frame.rows()/(double)screen_height;
	    		double x_offset = (scale*(double)screen_width-(double)input_frame.cols())/2;
	    		int x_modified = (int)((double)x*scale-x_offset);
	    		int y_modified = (int)((double)y*scale);
	    		mask_frame.put(y_modified,x_modified,1);
	    	}
	        break;
	        
	    case MotionEvent.ACTION_DOWN:
	    	
	    	break;
	        
	    case MotionEvent.ACTION_UP:
	    	
	    	// Required function call
	        v.performClick();
	        
	        // Reset back to the beginning!
	        if (inpainted_configured==true||inpainted_captured==true) {
	        	
	        	inpainted_captured = false;
	        	inpainted_configured = false;
	        	overlap_captured = false;
	        	mask_captured = false;
	        	screen_captured = false;
	        	
	        // Initialize the algorithm
	        } else if (overlap_captured==true) {
	        	
	        	// The mask needs to be in its 8-bit format 
	        	mask_frame.convertTo(mask_frame_8u, CvType.CV_8U);
	        	
	        	// If this is the first time for inpainting,
	        	// initialize the necessary native code
	        	if (handle==0) {
	        		native_setup(
	        				captured_frame.getNativeObjAddr(),
	        				mask_frame_8u.getNativeObjAddr(),
	        				inpainted_frame.getNativeObjAddr());
	        	} 
	        	
	        	// Ensure the native software is set to its first mode of operation
	        	native_reset();
	        	
	        	// Determine the total numbers that need to be inpainted
	        	pixels_total = native_get_pixels();
	        	
	        	// Raise flag to indicate the native program has been configured 
	        	// and the inpainting can now commence
	        	inpainted_configured = true;
	        	
        	// Display the overlapped mask 
	        } else if (mask_captured==true) {
	        	
	        	// Set the display frame
	        	Mat zeros_frame = new Mat(input_frame.size(), CvType.CV_8U, Scalar.all(0));
	        	mask_frame.convertTo(mask_frame_8u, CvType.CV_8UC1);
	        	Core.compare(mask_frame_8u, zeros_frame, mask_frame_8u, Core.CMP_EQ);
	        	captured_frame.copyTo(display_frame,mask_frame_8u);
	        	Core.multiply(display_frame, new Scalar(255), display_frame);
	        	display_frame.convertTo(display_frame, CvType.CV_8UC1);
	        	zeros_frame.release();

	        	// Set this flag to indicate the overlap image has been updated
	        	overlap_captured = true;
	        	
	        // Go into mask capture mode in order to determine
	        // the mask frame
	    	} else if (screen_captured==true) {
	        	
	    		// Apply dilation
	    		Imgproc.dilate(mask_frame, mask_frame, dilate_frame);
	    		
	    		// Set the display frame
	    		Core.multiply(mask_frame, Scalar.all(255), display_frame);
	    		display_frame.convertTo(display_frame, CvType.CV_8UC1);
	    		
	        	// Set the flag to indicate the mask has been determined
	        	mask_captured = true;
	        	
        	// Go into screen captured state once the screen is 
        	// touched and an input frame is available
	        } else if (input_frame!=null) {
	        	
	        	// Capture input frame
	        	input_frame.convertTo(captured_frame,CvType.CV_32FC1);
	        	Core.divide(captured_frame, new Scalar(255), captured_frame);
	        	input_frame.copyTo(display_frame);
	        	
	        	// Set up the mask frame
	        	mask_frame.create(input_frame.size(),CvType.CV_32FC1);
	        	mask_frame.setTo(Scalar.all(0));
	        	
	        	// Set the flag that indicates the program the screen has been captured
	        	screen_captured = true;
	        }
	        break;
	        
	    default:
	        break;
	    }
	    return true;
	}
}
