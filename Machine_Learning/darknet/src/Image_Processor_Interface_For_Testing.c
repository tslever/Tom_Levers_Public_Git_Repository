#include "darknet.h"
#include "Image_Processor_Interface_For_Testing.h"


image DNLIB_load_image_color_for_testing(char* filename, int w, int h) {

    printf("Testing DNLIB_load_image_color_for_testing.\n");

    image* The_Pointer_To_The_Image = (image*)malloc(sizeof(image));
    return *The_Pointer_To_The_Image;

}


image DNLIB_resize_min_for_testing(image im, int min) {

    printf("Testing_DNLIB_resize_min_for_testing.\n");

    image* The_Pointer_To_The_Image = (image*)malloc(sizeof(image));
    return *The_Pointer_To_The_Image;

}


image DNLIB_crop_image_for_testing(image im, int dx, int dy, int w, int h) {

    printf("Testing_DNLIB_crop_image_for_testing.\n");

    image* The_Pointer_To_The_Image = (image*)malloc(sizeof(image));
    return *The_Pointer_To_The_Image;

}