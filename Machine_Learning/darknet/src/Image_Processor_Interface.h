#ifndef IMAGE_PROCESSOR_INTERFACE_H
#define IMAGE_PROCESSOR_INTERFACE_H


#include "darknet.h"


image DNLIB_load_image_color(char* filename, int w, int h);
image DNLIB_resize_min(image im, int min);
image DNLIB_crop_image(image im, int dx, int dy, int w, int h);


#endif