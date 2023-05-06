#ifndef AN_IMAGE_PROCESSOR_INTERFACE_H
#define AN_IMAGE_PROCESSOR_INTERFACE_H


/* When the AN_IMAGE_INTERFACE_EXPORTS macro is defined, the
 * AN_IMAGE_INTERFACE_API macro sets the __declspec(dllexport) modifier on the
 * function declarations. This modifier tells the compiler and linker to export a function
 * or variable from the DLL for use by other applications. When
 * AN_IMAGE_INTERFACE_EXPORTS is undefined, for example, when the header file
 * is included by a client application, AN_IMAGE_INTERFACE_API applies the
 * __declspec(dllimport) modifier to the declarations. This modifier optimizes the import
 * of the function or variable in an application.
 * https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-160
 */
#ifdef AN_IMAGE_PROCESSOR_INTERFACE_EXPORTS
#define AN_IMAGE_PROCESSOR_INTERFACE_API __declspec(dllexport)
#else
#define AN_IMAGE_PROCESSOR_INTERFACE_API __declspec(dllimport)
#endif


#include "darknet.h"
//#include <opencv2\opencv.hpp>


//AN_IMAGE_PROCESSOR_INTERFACE_API image DNLIB_load_image_color(char* filename, int w, int h);
//AN_IMAGE_PROCESSOR_INTERFACE_API image DNLIB_resize_min(image im, int min);
//AN_IMAGE_PROCESSOR_INTERFACE_API image DNLIB_crop_image(image im, int dx, int dy, int w, int h);

AN_IMAGE_PROCESSOR_INTERFACE_API float* DNLIB_provides_a_pointer_to_the_data_from_a_loaded_resized_and_cropped_image_based_on(char* the_name_of_the_image_file, network the_neural_network);
//AN_IMAGE_PROCESSOR_INTERFACE_API float* DNLIB_provides_a_pointer_to_the_data_from_a_loaded_resized_and_cropped_image_based_on(cv::Mat the_image, network the_neural_network);


#endif