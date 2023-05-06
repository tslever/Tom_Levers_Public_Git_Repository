#ifndef A_WEIGHTS_FILE_PROCESSOR_INTERFACE_H
#define A_WEIGHTS_FILE_PROCESSOR_INTERFACE_H


/* When the A_WEIGHTS_FILE_INTERFACE_EXPORTS macro is defined, the
 * A_WEIGHTS_FILE_INTERFACE_API macro sets the __declspec(dllexport) modifier on the
 * function declarations. This modifier tells the compiler and linker to export a function
 * or variable from the DLL for use by other applications. When
 * A_WEIGHTS_FILE_INTERFACE_EXPORTS is undefined, for example, when the header file
 * is included by a client application, A_WEIGHTS_FILE_INTERFACE_API applies the
 * __declspec(dllimport) modifier to the declarations. This modifier optimizes the import
 * of the function or variable in an application.
 * https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-160
 */
#ifdef A_WEIGHTS_FILE_PROCESSOR_INTERFACE_EXPORTS
#define A_WEIGHTS_FILE_PROCESSOR_INTERFACE_API __declspec(dllexport)
#else
#define A_WEIGHTS_FILE_PROCESSOR_INTERFACE_API __declspec(dllimport)
#endif


#include "darknet.h"


//A_WEIGHTS_FILE_PROCESSOR_INTERFACE_API network DNLIB_parse_network_cfg_custom(char* filename, int batch, int time_steps);
//A_WEIGHTS_FILE_PROCESSOR_INTERFACE_API void DNLIB_load_weights(network* net, char* filename);
A_WEIGHTS_FILE_PROCESSOR_INTERFACE_API network DNLIB_provides_a_neural_network_with_loaded_weights_based_on(char* the_name_of_the_network_configuration_file, char* the_name_of_the_weights_file);


#endif