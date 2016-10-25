LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# OpenCV-related
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
include C:\OpenCV-android-sdk\sdk\native\jni\OpenCv.mk

LOCAL_MODULE    := native
LOCAL_SRC_FILES := native.cpp
LOCAL_SRC_FILES += inpainting_algorithms.cpp
LOCAL_LDLIBS    := -llog
include $(BUILD_SHARED_LIBRARY)



