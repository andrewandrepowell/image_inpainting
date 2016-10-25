#include "com_example_native_project_MainActivity.h"
#include "inpainting_algorithms.h"
#include <memory>

// Bloggers are heros deserving of all the accolades!!!
// http://www.drdobbs.com/cpp/accessing-c11-features-of-the-android-nd/240168385?pgno=2

using namespace std;
using namespace cv;
using namespace inpainting_algorithms;

static void set_handle(JNIEnv *env, jobject this_, void* handle);
static void* get_handle(JNIEnv *env, jobject this_);

class Native {
public:
	Native(Mat& input_array, Mat& mask_array, Mat& output_array) :
		input_array(input_array), mask_array(mask_array), output_array(output_array),
		patch_size_0(30),distance_size(patch_size_0*1),skip_factor(1,1),
		cahn_epsilons{1},
		cahn_total_iters{50},
		inpainting_mode(PROPOSED)  { }
	~Native() { }
	void reset() { inpainting_mode = PROPOSED; }
	int get_pixels() {
		Inpainting_Base_0* ptr = inpainting_ptr.get();
		if ((ptr!=NULL)&&
				(typeid(*ptr)==typeid(Proposed_inpainting_0))) {
			return ((Proposed_inpainting_0*)ptr)->get_pixels();
		} else {
			return 0;
		}
	}
	bool perform() {
		if (inpainting_ptr.get()==NULL) {
			switch (inpainting_mode) {
			case PROPOSED:
				inpainting_ptr.reset(new Proposed_inpainting_0(
						input_array,mask_array,output_array,
					patch_size_0,distance_size,skip_factor));
				inpainting_mode = CAHN;
				break;
			case CAHN:
				inpainting_ptr.reset(new Cahn_hilliard_gillette_inpainting_0(
						output_array,mask_array,output_array,
					cahn_total_iters,cahn_epsilons,total_stages));
				inpainting_mode = FINISHED;
				break;
			case FINISHED:
				return true;
			}
		} else if (inpainting_ptr->perform()==true) {
			inpainting_ptr.reset();
		}
		return false;
	}
	void update() {
		Inpainting_Base_0* ptr = inpainting_ptr.get();
		if (ptr!=NULL) {
			ptr->update();
		}
	}
private:
	const static int total_stages = 1;
	int patch_size_0;
	int distance_size;
	Size skip_factor;
	float cahn_epsilons[total_stages];
	int cahn_total_iters[total_stages];
	Mat& input_array;
	Mat& mask_array;
	Mat& output_array;
	unique_ptr<Inpainting_Base_0> inpainting_ptr;
	enum { PROPOSED, CAHN, FINISHED } inpainting_mode;
};

void set_handle(JNIEnv *env, jobject this_, void* handle) {
	env->SetLongField(
		this_,
		env->GetFieldID(env->GetObjectClass(this_), "handle", "J"),
		(jlong)handle);
}

void* get_handle(JNIEnv *env, jobject this_) {
	return (void*)env->GetLongField(this_,
		env->GetFieldID(env->GetObjectClass(this_),
		"handle", "J"));
}

JNIEXPORT void JNICALL Java_com_example_native_1project_MainActivity_native_1setup
  (JNIEnv *env, jobject this_, jlong input_array_addr, jlong mask_array_addr, jlong output_array_addr) {

	Mat* input_array_ptr = (Mat*)input_array_addr;
	Mat* mask_array_ptr = (Mat*)mask_array_addr;
	Mat* output_array_ptr = (Mat*)output_array_addr;

	set_handle(env,this_,new Native(*input_array_ptr,*mask_array_ptr,*output_array_ptr));
}

JNIEXPORT void JNICALL Java_com_example_native_1project_MainActivity_native_1reset
  (JNIEnv *env, jobject this_) {
	((Native*)get_handle(env,this_))->reset();
}

JNIEXPORT void JNICALL Java_com_example_native_1project_MainActivity_native_1destroy
  (JNIEnv *env, jobject this_) {
	delete ((Native*)get_handle(env,this_));
	set_handle(env,this_,NULL);
}

JNIEXPORT jint JNICALL Java_com_example_native_1project_MainActivity_native_1get_1pixels
  (JNIEnv *env, jobject this_) {
	return ((Native*)get_handle(env,this_))->get_pixels();
}

JNIEXPORT jboolean JNICALL Java_com_example_native_1project_MainActivity_native_1perform
  (JNIEnv *env, jobject this_) {
	return (jboolean)((Native*)get_handle(env,this_))->perform();
}

JNIEXPORT void JNICALL Java_com_example_native_1project_MainActivity_native_1update
  (JNIEnv *env, jobject this_) {
	((Native*)get_handle(env,this_))->update();
}

JNIEXPORT void JNICALL Java_com_example_native_1project_MainActivity_perform_1proposed_1inpainting_10(JNIEnv *env, jobject obj,
		  jlong input_array_addr, jlong mask_array_addr,
		  jlong output_array_addr) {

	static int patch_size_0 = 30;
	static int distance_size = patch_size_0*1;
	static Size skip_factor(1,1);
	Mat* input_array_ptr = (Mat*)input_array_addr;
	Mat* mask_array_ptr = (Mat*)mask_array_addr;
	Mat* output_array_ptr = (Mat*)output_array_addr;

	perform_proposed_inpainting_0(
			*input_array_ptr,*mask_array_ptr,*output_array_ptr,
			patch_size_0,distance_size,
			skip_factor);
}


