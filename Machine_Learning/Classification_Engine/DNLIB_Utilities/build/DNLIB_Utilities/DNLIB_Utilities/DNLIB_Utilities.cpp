#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include "DNLIB_Utilities.h"


void calloc_error()
{
    fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}


int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}


void* xcalloc(size_t nmemb, size_t size) {
    void* ptr = calloc(nmemb, size);
    if (!ptr) {
        calloc_error();
    }
    return ptr;
}