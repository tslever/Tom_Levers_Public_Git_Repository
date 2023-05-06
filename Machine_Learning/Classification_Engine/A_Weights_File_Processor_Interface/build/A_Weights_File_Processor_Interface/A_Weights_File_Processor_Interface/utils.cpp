#include "pch.h"
#include "utils.h"
#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include <corecrt_math.h>


void file_error(char* s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(EXIT_FAILURE);
}


void error(const char* s)
{
    perror(s);
    assert(0);
    exit(EXIT_FAILURE);
}


void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        malloc_error();
    }
    return ptr;
}


void malloc_error()
{
    fprintf(stderr, "xMalloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}


char* fgetl(FILE* fp)
{
    if (feof(fp)) return 0;
    size_t size = 512;
    char* line = (char*)xmalloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = (char*)xrealloc(line, size * sizeof(char));
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX) readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (curr >= 2)
        if (line[curr - 2] == 0x0d) line[curr - 2] = 0x00;

    if (curr >= 1)
        if (line[curr - 1] == 0x0a) line[curr - 1] = 0x00;

    return line;
}


void* xrealloc(void* ptr, size_t size) {
    ptr = realloc(ptr, size);
    if (!ptr) {
        realloc_error();
    }
    return ptr;
}


void realloc_error()
{
    fprintf(stderr, "Realloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}


void strip(char* s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0d || c == 0x0a) ++offset;
        else s[i - offset] = c;
    }
    s[len - offset] = '\0';
}


float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }

#if (RAND_MAX < 65536)
    int rnd = rand() * (RAND_MAX + 1) + rand();
    return ((float)rnd / (RAND_MAX * RAND_MAX) * (max - min)) + min;
#else
    return ((float)rand() / RAND_MAX * (max - min)) + min;
#endif
    //return (random_float() * (max - min)) + min;
}


float sum_array(float* a, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) sum += a[i];
    return sum;
}


float mag_array(float* a, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}


int* read_map(char* filename)
{
    int n = 0;
    int* map = 0;
    char* str;
    FILE* file;// = fopen(filename, "r");
    fopen_s(&file, filename, "r");
    if (!file) file_error(filename);
    while ((str = fgetl(file))) {
        ++n;
        map = (int*)xrealloc(map, n * sizeof(int));
        map[n - 1] = atoi(str);
        free(str);
    }
    if (file) fclose(file);
    return map;
}


int int_index(int* a, int val, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (a[i] == val) return i;
    }
    return -1;
}


void top_k(float* a, int n, int k, int* index)
{
    int i, j;
    for (j = 0; j < k; ++j) index[j] = -1;
    for (i = 0; i < n; ++i) {
        int curr = i;
        for (j = 0; j < k; ++j) {
            if ((index[j] < 0) || a[curr] > a[index[j]]) {
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}