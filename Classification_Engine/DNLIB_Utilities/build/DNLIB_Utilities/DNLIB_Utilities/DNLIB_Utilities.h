#ifndef DNLIB_UTILITIES_H
#define DNLIB_UTILITIES_H


/* When the DNLIB_UTILITIES_EXPORTS macro is defined, the
 * DNLIB_UTILITIES_API macro sets the __declspec(dllexport) modifier on the
 * function declarations. This modifier tells the compiler and linker to export a function
 * or variable from the DLL for use by other applications. When
 * DNLIB_EXPORTS is undefined, for example, when the header file
 * is included by a client application, DNLIB_API applies the
 * __declspec(dllimport) modifier to the declarations. This modifier optimizes the import
 * of the function or variable in an application.
 * https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-160
 */
#ifdef DNLIB_UTILITIES_EXPORTS
#define DNLIB_UTILITIES_API __declspec(dllexport)
#else
#define DNLIB_UTILITIES_API __declspec(dllimport)
#endif


DNLIB_UTILITIES_API void calloc_error();
DNLIB_UTILITIES_API int constrain_int(int a, int min, int max);
DNLIB_UTILITIES_API void* xcalloc(size_t nmemb, size_t size);


#endif