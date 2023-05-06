#ifndef A_CLASSIFICATION_ENGINE_INTERFACE_H
#define A_CLASSIFICATION_ENGINE_INTERFACE_H


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
#ifdef A_CLASSIFICATION_ENGINE_INTERFACE_EXPORTS
#define A_CLASSIFICATION_ENGINE_INTERFACE_API __declspec(dllexport)
#else
#define A_CLASSIFICATION_ENGINE_INTERFACE_API __declspec(dllimport)
#endif


#include "darknet.h"


A_CLASSIFICATION_ENGINE_INTERFACE_API float* DNLIB_network_predict(network net, float* input);
A_CLASSIFICATION_ENGINE_INTERFACE_API void DNLIB_hierarchy_predictions(float* predictions, int n, tree* hier, int only_leaves);
A_CLASSIFICATION_ENGINE_INTERFACE_API void DNLIB_top_k(float* a, int n, int k, int* index);


#endif