#ifndef STD_INT
#define STD_INT
#include <stdint.h>
#endif

__global__
void ConvertFloatDataToIntData(int16_t* intDataOfAudioFile, float* floatDataOfAudioFile, float maximumMagnitudeOfConvolvedAudioFile)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    intDataOfAudioFile[index] = (int16_t)(floatDataOfAudioFile[index] / maximumMagnitudeOfConvolvedAudioFile * 32767 /*= pow(2.0, 15.0) - 1.0*/);
}