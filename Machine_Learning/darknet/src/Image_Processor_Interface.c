#include "darknet.h"
#include "Image_Processor_Interface.h"


image DNLIB_load_image_color(char* filename, int w, int h) {

    return load_image_color(filename, w, h);

}


image DNLIB_resize_min(image im, int min) {

    return resize_min(im, min);

}


image DNLIB_crop_image(image im, int dx, int dy, int w, int h) {

    return crop_image(im, dx, dy, w, h);

}