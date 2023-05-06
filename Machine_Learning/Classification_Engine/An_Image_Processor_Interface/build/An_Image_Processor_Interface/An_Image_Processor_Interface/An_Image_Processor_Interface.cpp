#include "pch.h" // Must be the first inclusion
#include "An_Image_Processor_Interface.h"


//image DNLIB_load_image_color(char* filename, int w, int h) {
//
//	printf("Testing DNLIB_load_image_color.\n");
//	image* thePointerToAnImage = (image*)malloc(sizeof(image));
//	return *thePointerToAnImage;
//
//	//return load_image_color(char* filename, int w, int h);
//
//}


//image DNLIB_resize_min(image im, int min) {
//
//	printf("Testing DNLIB_resize_min.\n");
//	image* thePointerToAnImage = (image*)malloc(sizeof(image));
//	return *thePointerToAnImage;
//
//	//return resize_min(im, min);
//
//}


//image DNLIB_crop_image(image im, int dx, int dy, int w, int h) {
//
//	printf("Testing DNLIB_crop_image.\n");
//	image* thePointerToAnImage = (image*)malloc(sizeof(image));
//	return *thePointerToAnImage;
//
//	//return crop_image(im, dx, dy, w, h);
//
//}


float* DNLIB_provides_a_pointer_to_the_data_from_a_loaded_resized_and_cropped_image_based_on(char* the_name_of_the_image_file, network the_neural_network)
{
	image im = load_image_color(the_name_of_the_image_file, 0, 0);
	image resized = resize_min(im, the_neural_network.w);
	image cropped = crop_image(resized, (resized.w - the_neural_network.w) / 2, (resized.h - the_neural_network.h) / 2, the_neural_network.w, the_neural_network.h);

	return cropped.data;
}