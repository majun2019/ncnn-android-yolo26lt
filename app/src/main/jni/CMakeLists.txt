project(yolo26ncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolo26ncnn SHARED yolo26ncnn.cpp yolo26.cpp yolo26_det.cpp yolo26_seg.cpp yolo26_pose.cpp yolo26_cls.cpp yolo26_obb.cpp ndkcamera.cpp)

target_link_libraries(yolo26ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
