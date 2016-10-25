#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include "..\inpainting_algorithms\inpainting_algorithms.h"

using namespace inpainting_algorithms;
using namespace std;
using namespace cv;

/* Data necesssary for the demonstration, barring parameters specific
   to the inpainting algorithm */
static char* window_name = "Demonstration Window";
static char* occlusion_name = "Occlusion Window";
static char* inpainted_name = "Inpainted Window";
static Mat captured_raw_frame;
static Mat occlusion_mask;
static Mat inpainted_image;
static Point start_point;
static Point prev_point;
static Point next_point;
static Point seed_point;
static int flood_points;
static bool left_button_clicked = false;
static bool right_button_clicked = false;
static bool screen_captured = false;
static bool mask_captured = false;
static bool proposed_inpaint_finished = false;

/* Parameters for the inpainting algorithm */
static int patch_size_0=20;
static int distance_size = patch_size_0*5;
static Size skip_factor(1,1);
static float cahn_epsilons[] = {1};
static int cahn_total_iters[] = {50};
static int total_stages = 1;
static unique_ptr<Inpainting_Base_0> inpainting_ptr;
static enum { PROPOSED, CAHN, FINISHED } inpainting_mode;

void mouse_handler(int event, int x, int y, int flags, void* parameter) {

	switch (event) {

	case EVENT_LBUTTONDOWN: 

		/* Screen captured mode */
		screen_captured ^= true;

		/* Reset all flags if user turns of captured screen */
		if (screen_captured==false) {
			mask_captured = false;
			proposed_inpaint_finished = false;
		}
		break;

	case EVENT_LBUTTONUP: 
		break;

	case EVENT_RBUTTONDOWN: 

		/* Actions performed only in screen captured mode */
		if (screen_captured==true) {

			/* Initialize the creation of the occlusion mask */
			start_point = cv::Point(x,y);
			seed_point = start_point;
			prev_point = start_point;
			flood_points = 1;
			occlusion_mask = Mat(captured_raw_frame.size(),CV_8U,Scalar(0));
		}
		break;

	case cv::EVENT_RBUTTONUP: 

		/* Actions performed only in screen captured mode */
		if (screen_captured==true) {

			/* Connect occlusion mask */
			line(occlusion_mask,prev_point,start_point,Scalar(1));

			/* Perform flood filling in occlusion */
			seed_point.x /= flood_points;
			seed_point.y /= flood_points;
			floodFill(occlusion_mask,seed_point,Scalar(1));

			/* Display window with the occlusion mask */
			Mat occlusion_mask_normalized = occlusion_mask==1;
			imshow(occlusion_name,occlusion_mask_normalized);

			/* Make sure the flags are reset */
			proposed_inpaint_finished = false;

			/* Start inpainting algorithm */
			inpainting_mode = PROPOSED;

			/* Go into the mask captured state to begin algorithm */
			mask_captured = true;
		}
		break;

	default: break;
	}

	switch (flags) {

	case EVENT_FLAG_LBUTTON:
		break;

	case EVENT_FLAG_RBUTTON:

		/* Make sure lines are connected */
		next_point = cv::Point(x,y);
		line(occlusion_mask,prev_point,next_point,Scalar(1));
		prev_point = next_point;

		/* Determine seed point for flood filled algorithm */
		seed_point += prev_point;
		flood_points++;
		break;

	default: break;
	}

}

int main(int argc, char* argv[]) {

	//try {
		/* Set up some stuff */
		namedWindow(occlusion_name);
		namedWindow(window_name);
		namedWindow(inpainted_name);
		setMouseCallback(window_name,mouse_handler,NULL);

		/* Initiate video */
		cout << "Enter id of video: ";
		int video_id;
		while (!(cin >> video_id)) {

			/* Just in case in correct letters accidently 
			   inputted to the console input stream */
			cin.clear();
			cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
		}
		VideoCapture video_capture(video_id);

		/*-- Main loop --*/
		while (true) {

			/* Required by OpenCV (and this is what calls the mouse_handler */
			waitKey(1);

			if (proposed_inpaint_finished==true) {
				
			/* Mask captured state */
			} else if (mask_captured==true) {

				/* Perform inpainting */
				if (inpainting_ptr.get()==NULL) {
					switch (inpainting_mode) {
					case PROPOSED:
						inpainting_ptr.reset(new Proposed_inpainting_0(
							captured_raw_frame,occlusion_mask,inpainted_image,
							patch_size_0,distance_size,skip_factor));
						inpainting_mode = CAHN;
						break;
					case CAHN:
						inpainting_ptr.reset(new Cahn_hilliard_gillette_inpainting_0(
							inpainted_image,occlusion_mask,inpainted_image,
							cahn_total_iters,cahn_epsilons,total_stages));
						inpainting_mode = FINISHED;
						break;
					case FINISHED:
						proposed_inpaint_finished = true;
						cout << "finished " << endl;
						break;
					}
				} else if (inpainting_ptr->perform()==false) {
					inpainting_ptr->update();
					imshow(inpainted_name,inpainted_image);
					if (typeid(*inpainting_ptr.get())==typeid(Proposed_inpainting_0)) {
						int iters = ((Proposed_inpainting_0*)inpainting_ptr.get())->get_pixels();
						cout << "Pixels remaining: " << iters << endl;
					}
				} else {
					inpainting_ptr.reset();
				}

			/* Screen captured satate */
			} else if (screen_captured==true) {

			/* Live feed state */
			} else {

				/* First, capture raw frame from camera */
				video_capture >> captured_raw_frame;

				/* For simplicity, let's convert the captured from into grayscale
				   and normalize it for the inpainting algorithm */
				cvtColor(captured_raw_frame,captured_raw_frame,CV_BGR2GRAY);
				captured_raw_frame.convertTo(captured_raw_frame,CV_32FC1);
				divide(captured_raw_frame,Scalar::all(255),captured_raw_frame);

				/* Update the window that displays the captured raw frames */
				imshow(window_name,captured_raw_frame);
			}
		}

	/* Report any errors */
	//} catch (exception& e) { cout << e.what() << endl; }

	/* Prevent program from closing, immediately */
	cin.ignore(numeric_limits<streamsize>::max(), '\n');
	return 0;
}