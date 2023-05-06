#ifndef IMAGE_H
#define IMAGE_H


#include "darknet.h"


image copy_image(image p);
image load_image(char* filename, int w, int h, int c);
image make_empty_image(int w, int h, int c);


#endif