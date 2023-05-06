#ifndef CLASSIFICATION_ENGINE_INTERFACE_FOR_TESTING_H
#define CLASSIFICATION_ENGINE_INTERFACE_FOR_TESTING_H


/* When the CLASSIFICATION_ENGINE_INTERFACE_EXPORTS macro is defined, the
 * CLASSIFICATION_ENGINE_INTERFACE_API macro sets the __declspec(dllexport) modifier on the
 * function declarations. This modifier tells the compiler and linker to export a function
 * or variable from the DLL for use by other applications. When
 * CLASSIFICATION_ENGINE_INTERFACE_EXPORTS is undefined, for example, when the header file
 * is included by a client application, CLASSIFICATION_ENGINE_INTERFACE_API applies the
 * __declspec(dllimport) modifier to the declarations. This modifier optimizes the import
 * of the function or variable in an application.
 * https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-160
 */
#ifdef CLASSIFICATION_ENGINE_INTERFACE_EXPORTS
#define CLASSIFICATION_ENGINE_INTERFACE_API __declspec(dllexport)
#else
#define CLASSIFICATION_ENGINE_INTERFACE_API __declspec(dllimport)
#endif


#include "darknet.h"


CLASSIFICATION_ENGINE_INTERFACE_API float* DNLIB_network_predict_for_testing(network net, float* input);
CLASSIFICATION_ENGINE_INTERFACE_API void DNLIB_hierarchy_predictions_for_testing(float* predictions, int n, tree* hier, int only_leaves);
CLASSIFICATION_ENGINE_INTERFACE_API void DNLIB_top_k_for_testing(float* a, int n, int k, int* index);


#endif