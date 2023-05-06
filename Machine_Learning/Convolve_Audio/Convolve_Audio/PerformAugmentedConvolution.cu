__global__
void PerformAugmentedConvolution(
    float* convolvedHeyHoToTheGreenwood_child, float* floatDataOfHeyHoToTheGreenwood_child, float* floatDataOfImpulseResponse_child, long lengthOfFloatDataOfImpulseResponse_child)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + lengthOfFloatDataOfImpulseResponse_child;

    convolvedHeyHoToTheGreenwood_child[index - lengthOfFloatDataOfImpulseResponse_child] = floatDataOfHeyHoToTheGreenwood_child[index];

    float summation = 0.0;

    int indexOfLastElementOfHeyHoToTheGreenwoodUnderImpulseResponse = index - 1;

    for (int i = 0; i < lengthOfFloatDataOfImpulseResponse_child; ++i) {
        summation += (floatDataOfImpulseResponse_child[i] * floatDataOfHeyHoToTheGreenwood_child[indexOfLastElementOfHeyHoToTheGreenwoodUnderImpulseResponse - i]);
    }

    summation *= 0.1;

    convolvedHeyHoToTheGreenwood_child[index - lengthOfFloatDataOfImpulseResponse_child] += summation;
}