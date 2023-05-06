#ifndef IMAGE_PROCESSOR_INTERFACE_FOR_TESTING_H
#define IMAGE_PROCESSOR_INTERFACE_FOR_TESTING_H


#include "darknet.h"


image DNLIB_load_image_color_for_testing(char* filename, int w, int h);
image DNLIB_resize_min_for_testing(image im, int min);
image DNLIB_crop_image_for_testing(image im, int dx, int dy, int w, int h);


#endif