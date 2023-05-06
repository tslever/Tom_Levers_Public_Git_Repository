#ifndef UTILS_H
#define UTILS_H


#include <stdio.h>
#include "darknet.h"


#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))


void file_error(char* s);
void error(const char* s);
void* xmalloc(size_t size);
void malloc_error();
char* fgetl(FILE* fp);
void* xrealloc(void* ptr, size_t size);
void realloc_error();
void strip(char* s);
float rand_uniform(float min, float max);
float sum_array(float* a, int n);
float mag_array(float* a, int n);
int* read_map(char* filename);
int int_index(int* a, int val, int n);


#endif