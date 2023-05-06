#include "cuda_runtime.h"

__global__
void ConvertFloatDataToIntData(int16_t* intDataOfAudioFile, float* floatDataOfAudioFile, float maximumMagnitudeOfConvolvedAudioFile);