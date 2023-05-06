#ifndef STD_INT
#define STD_INT
#include <stdint.h>
#endif

__global__
void ConvertIntDataToFloatData(float* floatDataOfAudioFile, int16_t* intDataOfAudioFile)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    floatDataOfAudioFile[index] = (float)intDataOfAudioFile[index];
}