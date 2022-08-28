#include "cuda_runtime.h"

__global__
void ConvertIntDataToFloatData(float* floatDataOfAudioFile, int16_t* intDataOfAudioFile);