#ifndef IMAGE_OPENCV_H
#define IMAGE_OPENCV_H


#include "darknet.h"
#include <opencv2/opencv.hpp>


typedef void* mat_cv;

image load_image_cv(char* filename, int channels);
mat_cv* load_image_mat_cv(const char* filename, int flag);
image mat_to_image(cv::Mat mat);


#endif