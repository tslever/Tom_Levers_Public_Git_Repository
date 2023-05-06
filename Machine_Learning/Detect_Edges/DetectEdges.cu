
// Allow creating vectors of filenames of training images and encodings.
#include <vector>

// Allow creating elements of vectors.
#include <string>

// Allow use of GetFilenamesOfTrainingImages.
#include "GetFilenamesOfTrainingImages.h"

// Allow use of GetFilenamesOfEncodingsForTrainingImages.
#include "GetFilenamesOfEncodingsForTrainingImages.h"

// Allow use of GetInputTensorOnHost.
#include "GetInputTensorOnHost.h"

// Allow
// - creating cudnnHandle,
// - creating inputTensorDescriptor,
// - creating filterTensorDescriptor,
#include <cudnn.h>

#include "DisplayImagesInActivationOutputTensor.h"

// --------------------------------------
// Allows use of std::cout and std::endl.
#include <iostream>
// --------------------------------------

int main()
{
	// -------------------------------------------------------------
	// Define vectors of filenames of training images and encodings.
	// -------------------------------------------------------------
	// Define a path to training images and encodings.
	std::string pathToImages = "Images\\";

	// Define a vector of filenames of training images.
	std::vector<std::string> filenamesOfTrainingImages = GetFilenamesOfTrainingImages(pathToImages);

	// -------------------
	// Create cudnnHandle.
	// -------------------
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);

	// ------------------------
	// Define inputTensorOnGPU.
	// ------------------------
	// Define properties of inputTensor.
	int imagesInSubdivision = 4;
	int channelsInImage = 3;
	int heightOfImage = 416;
	int widthOfImage = 416;
	int elementsInImage = channelsInImage * heightOfImage * widthOfImage;
	int elementsInInputTensor = imagesInSubdivision * channelsInImage * heightOfImage * widthOfImage;

	// Define inputTensorDescriptor.
	cudnnTensorDescriptor_t inputTensorDescriptor;
	cudnnCreateTensorDescriptor(&inputTensorDescriptor);
	cudnnSetTensor4dDescriptor(
		/*tensorDesc=*/inputTensorDescriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_DOUBLE,
		/*n=*/imagesInSubdivision,
		/*c=*/channelsInImage,
		/*h=*/heightOfImage,
		/*w=*/widthOfImage);

	// Declare a pointer for training images in host heap that will be used by GetInputTensorOnHost.
	// I must use a pointer to access image buffer.
	// I encounter a runtime error if I define and delete the pointer in GetInputTensorOnHost.
	double* pointerToTrainingImage = new double[elementsInImage];

	// Define a pointer to inputTensorOnHost in host heap.
	double* inputTensorOnHost = new double[elementsInInputTensor];
	GetInputTensorOnHost(
		inputTensorOnHost,
		pointerToTrainingImage,
		imagesInSubdivision,
		channelsInImage,
		heightOfImage,
		widthOfImage,
		pathToImages,
		filenamesOfTrainingImages);

	// Delete pointer for training images.
	delete pointerToTrainingImage;

	// Define a pointer to inputTensorOnGPU in GPU heap.
	double* inputTensorOnGPU;
	cudaMalloc(&inputTensorOnGPU, elementsInInputTensor * sizeof(double));
	cudaMemcpy(inputTensorOnGPU, inputTensorOnHost, elementsInInputTensor * sizeof(double), cudaMemcpyHostToDevice);

	// Delete pointer inputTensorOnHost.
	delete inputTensorOnHost;

	// -------------------------
	// Define filterTensorOnGPU.
	// -------------------------
	// Define properties of filterTensor.
	const int filters = 3;
	const int channelsInFilter = 3;
	const int heightOfFilter = 3;
	const int widthOfFilter = 3;
	const int elementsInFilterTensor = filters * channelsInFilter * heightOfFilter * widthOfFilter;

	// Define filterTensorDescriptor.
	cudnnFilterDescriptor_t filterTensorDescriptor;
	cudnnCreateFilterDescriptor(&filterTensorDescriptor);
	cudnnSetFilter4dDescriptor(
		/*filterDesc=*/filterTensorDescriptor,
		/*dataType=*/CUDNN_DATA_DOUBLE,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*k=*/filters,
		/*c=*/channelsInFilter,
		/*h=*/heightOfFilter,
		/*w=*/widthOfFilter
	);

	// Define pointer to filterTensorOnHost in host heap.
	double* filterTensorOnHost = new double[elementsInFilterTensor];
	double channelTemplate[3][3] = {
		{1.0,1.0,1.0},
		{1.0,-8.0,1.0},
		{1.0,1.0,1.0}
	};
	for (int f = 0; f < filters; ++f)
	{
		for (int c = 0; c < channelsInFilter; ++c)
		{
			for (int h = 0; h < heightOfFilter; ++h)
			{
				for (int w = 0; w < widthOfFilter; ++w)
				{
					filterTensorOnHost[w + h*widthOfFilter + c*heightOfFilter*widthOfFilter + f*channelsInFilter*heightOfFilter*widthOfFilter] =
						channelTemplate[h][w];
				}
			}
		}
	}

	// Define pointer to filterTensorOnGPU in GPU heap.
	double* filterTensorOnGPU;
	cudaMalloc(&filterTensorOnGPU, elementsInFilterTensor * sizeof(double));
	cudaMemcpy(filterTensorOnGPU, filterTensorOnHost, elementsInFilterTensor * sizeof(double), cudaMemcpyHostToDevice);

	// Delete pointer filterTensorOnHost.
	delete filterTensorOnHost;

	// --------------------
	// Perform convolution.
	// --------------------
	// Define convolutionHyperparametersDescriptor.
	cudnnConvolutionDescriptor_t convolutionHyperparametersDescriptor;
	cudnnCreateConvolutionDescriptor(&convolutionHyperparametersDescriptor);
	cudnnSetConvolution2dDescriptor(
		/*convDesc=*/convolutionHyperparametersDescriptor,
		/*pad_h=*/1,
		/*pad_w=*/1,
		/*u= vertical filter stride =*/1,
		/*v= horizontal filter stride =*/1,
		/*dilation_h=*/1,
		/*dilation_w=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_DOUBLE);

	// Define properties of outputTensor.
	int outputSubtensors = 0;
	int channelsInOutputSubtensor = 0;
	int heightOfOutputSubtensor = 0;
	int widthOfOutputSubtensor = 0;

	cudnnGetConvolution2dForwardOutputDim(
		/*convDesc=*/convolutionHyperparametersDescriptor,
		/*inputTensorDesc=*/inputTensorDescriptor,
		/*filterDesc=*/filterTensorDescriptor,
		/**n=*/&outputSubtensors,
		/**c=*/&channelsInOutputSubtensor,
		/**h=*/&heightOfOutputSubtensor,
		/**w*/&widthOfOutputSubtensor);

	int elementsInOutputTensor = outputSubtensors * channelsInOutputSubtensor * heightOfOutputSubtensor * widthOfOutputSubtensor;

	// Create outputTensorDescriptor.
	cudnnTensorDescriptor_t outputTensorDescriptor;
	cudnnCreateTensorDescriptor(&outputTensorDescriptor);
	cudnnSetTensor4dDescriptor(
		/*tensorDesc=*/outputTensorDescriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*datatType=*/CUDNN_DATA_DOUBLE,
		/*n=*/outputSubtensors,
		/*c=*/channelsInOutputSubtensor,
		/*h=*/heightOfOutputSubtensor,
		/*w=*/widthOfOutputSubtensor);

	// Define convolutionAlgorithmDescriptor.
	// After call to cudnnGetConvolutionForwardAlgorithm,
	// convolutionAlgorithm = 0 (CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM).
	cudnnConvolutionFwdAlgo_t convolutionAlgorithmDescriptor;
	cudnnGetConvolutionForwardAlgorithm(
		/*handle=*/cudnnHandle,
		/*xDesc=*/inputTensorDescriptor,
		/*wDesc=*/filterTensorDescriptor,
		/*convDesc=*/convolutionHyperparametersDescriptor,
		/*yDesc=*/outputTensorDescriptor,
		/*preference=*/CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		/*memoryLimitInBytes=*/0, // No limit.
		/**algo=*/&convolutionAlgorithmDescriptor);

	// Declare pointer to workspaceOnGPU in GPU heap.
	size_t bytesInWorkspace;

	cudnnGetConvolutionForwardWorkspaceSize(
		/*handle=*/cudnnHandle,
		/*xDesc=*/inputTensorDescriptor,
		/*wDesc=*/filterTensorDescriptor,
		/*convDesc=*/convolutionHyperparametersDescriptor,
		/*yDesc=*/outputTensorDescriptor,
		/*algo=*/convolutionAlgorithmDescriptor,
		/**sizeInBytes=*/&bytesInWorkspace);

	void* workspaceOnGPU;
	cudaMalloc(&workspaceOnGPU, bytesInWorkspace);

	// Define pointer to convolutionOutputTensorOnGPU in GPU heap.
	double* convolutionOutputTensorOnGPU;
	cudaMalloc(&convolutionOutputTensorOnGPU, elementsInOutputTensor * sizeof(double));

	// Perform convolution.
	double one = 1.0;
	double zero = 0.0;

	cudnnConvolutionForward(
		/*handle=*/cudnnHandle,
		/**alpha=*/&one,
		/*xDesc=*/inputTensorDescriptor,
		/**x=*/inputTensorOnGPU,
		/*wDesc=*/filterTensorDescriptor,
		/*w=*/filterTensorOnGPU,
		/*convDesc=*/convolutionHyperparametersDescriptor,
		/*algo=*/convolutionAlgorithmDescriptor,
		/*workSpace=*/workspaceOnGPU,
		/*workSpaceSizeInBytes=*/bytesInWorkspace,
		/**beta=*/&zero,
		/*yDesc=*/outputTensorDescriptor,
		/**y=*/convolutionOutputTensorOnGPU);

	// Free inputTensorOnGPU.
	cudaFree(inputTensorOnGPU);

	// Free filterTensorOnGPU.
	cudaFree(filterTensorOnGPU);

	// Free workspaceOnGPU.
	cudaFree(workspaceOnGPU);

	// ----------------------------
	// Perform batch normalization.
	// ----------------------------
	// Define pointer to batchNormOutputTensorOnGPU in GPU heap.
	double* batchNormOutputTensorOnGPU;
	cudaMalloc(&batchNormOutputTensorOnGPU, elementsInOutputTensor * sizeof(double));

	// Define batchNormMode.
	cudnnBatchNormMode_t batchNormMode = CUDNN_BATCHNORM_SPATIAL;

	// DeclareDefine derivedBNTensorDescriptor.
	cudnnTensorDescriptor_t derivedBNTensorDescriptor;
	cudnnCreateTensorDescriptor(&derivedBNTensorDescriptor);
	cudnnDeriveBNTensorDescriptor(
		/*derivedBNDesc=*/derivedBNTensorDescriptor,
		/*xDesc=*/outputTensorDescriptor,
		/*mode=*/batchNormMode);

	// Define pointer to batchNormScalesOnGPU in GPU heap.
	double* batchNormScalesOnHost = new double[channelsInOutputSubtensor];
	int c;
	for (c = 0; c < channelsInOutputSubtensor; ++c) {
		batchNormScalesOnHost[c] = 1.0;
	}
	double* batchNormScalesOnGPU;
	cudaMalloc(&batchNormScalesOnGPU, channelsInOutputSubtensor * sizeof(double));
	cudaMemcpy(batchNormScalesOnGPU, batchNormScalesOnHost, channelsInOutputSubtensor * sizeof(double), cudaMemcpyHostToDevice);
	delete batchNormScalesOnHost;

	// Define pointer to batchNormBiasesOnGPU in GPU heap.
	double* batchNormBiasesOnHost = new double[channelsInOutputSubtensor];
	for (c = 0; c < channelsInOutputSubtensor; ++c) {
		batchNormBiasesOnHost[c] = 0.0;
	}
	double* batchNormBiasesOnGPU;
	cudaMalloc(&batchNormBiasesOnGPU, channelsInOutputSubtensor * sizeof(double));
	cudaMemcpy(batchNormBiasesOnGPU, batchNormBiasesOnHost, channelsInOutputSubtensor * sizeof(double), cudaMemcpyHostToDevice);
	delete batchNormBiasesOnHost;

	// Define Exponential Average Factor. 
	double expAverageFactor = 1.0;

	// Define pointer to resultRunningMeansOnGPU in GPU heap.
	double* resultRunningMeansOnGPU;
	cudaMalloc(&resultRunningMeansOnGPU, channelsInOutputSubtensor * sizeof(double));

	// Define pointer to resultRunningVariancesOnGPU in GPU heap.
	double* resultRunningVariancesOnGPU;
	cudaMalloc(&resultRunningVariancesOnGPU, channelsInOutputSubtensor * sizeof(double));

	// Define epsilon.
	double epsln = 0.00001;

	// Define pointer to resultSaveMeanOnGPU in GPU heap.
	double* resultSaveMeanOnGPU;
	cudaMalloc(&resultSaveMeanOnGPU, channelsInOutputSubtensor * sizeof(double));

	// Define pointer to resultSaveInvVarianceOnGPU in GPU heap.
	double* resultSaveInvVarianceOnGPU;
	cudaMalloc(&resultSaveInvVarianceOnGPU, channelsInOutputSubtensor * sizeof(double));

	cudnnBatchNormalizationForwardTraining(
		/*handle=*/cudnnHandle,
		/*mode=*/batchNormMode,
		/**alpha=*/&one,
		/**beta=*/&zero,
		/*xDesc=*/outputTensorDescriptor,
		/**x=*/convolutionOutputTensorOnGPU,
		/*yDesc=*/outputTensorDescriptor,
		/**y=*/batchNormOutputTensorOnGPU,
		/*bnScaleBiasMeanVarDesc=*/derivedBNTensorDescriptor,
		/*bnScaleData=*/batchNormScalesOnGPU,
		/*bnBiasData=*/batchNormBiasesOnGPU,
		/*exponentialAverageFactor=*/expAverageFactor,
		/*resultRunningMeanData=*/resultRunningMeansOnGPU,
		/*resultRunningVarianceData=*/resultRunningVariancesOnGPU,
		/*epsilon=*/epsln,
		/*resultSaveMean=*/resultSaveMeanOnGPU,
		/*resultSaveInvVariance=*/resultSaveInvVarianceOnGPU);

	// Free convolutionOutputTensorOnGPU.
	cudaFree(convolutionOutputTensorOnGPU);

	// Free batchNormScalesOnGPU.
	cudaFree(batchNormScalesOnGPU);

	// Free batchNormBiasesOnGPU.
	cudaFree(batchNormBiasesOnGPU);

	// -----------------------------
	// Perform nonlinear activation.
	// -----------------------------
	// Define activationDescriptor.
	cudnnActivationDescriptor_t activationDescriptor;
	cudnnCreateActivationDescriptor(&activationDescriptor);
	cudnnSetActivationDescriptor(
		/*activationDesc=*/activationDescriptor,
		/*mode=*/CUDNN_ACTIVATION_RELU,
		/*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
		/*coef=*/std::numeric_limits<double>::infinity());

	// Define pointer to activationOutputTensorOnGPU in GPU heap.
	double* activationOutputTensorOnGPU;
	cudaMalloc(&activationOutputTensorOnGPU, elementsInOutputTensor * sizeof(double));

	// Perform activation.
	cudnnActivationForward(
		/*handle=*/cudnnHandle,
		/*activationDesc=*/activationDescriptor,
		/**alpha=*/&one,
		/**xDesc=*/outputTensorDescriptor,
		/**x=*/batchNormOutputTensorOnGPU,
		/**beta=*/&zero,
		/*yDesc=*/outputTensorDescriptor,
		/**y=*/activationOutputTensorOnGPU);

	// Free batchNormOutputTensorOnGPU.
	cudaFree(batchNormOutputTensorOnGPU);

	// -------------------------------------------
	// Copy activationOutputTensor on GPU to host.
	// -------------------------------------------
	// Define pointer to activationOutputTensorOnHost in host heap.
	double* activationOutputTensorOnHost = new double[elementsInOutputTensor];
	cudaMemcpy(
		activationOutputTensorOnHost,
		activationOutputTensorOnGPU,
		elementsInOutputTensor * sizeof(double),
		cudaMemcpyDeviceToHost);

	cudaFree(activationOutputTensorOnGPU);

	// -----------------------------------------------
	// Display images in activationOutputTensorOnHost.
	// -----------------------------------------------
	DisplayImagesInActivationOutputTensor(
		activationOutputTensorOnHost,
		outputSubtensors,
		heightOfOutputSubtensor,
		widthOfOutputSubtensor,
		channelsInOutputSubtensor);

	// Free activationOutputTensorOnHost.
	delete activationOutputTensorOnHost;

	cudaFree(resultRunningMeansOnGPU);
	cudaFree(resultRunningVariancesOnGPU);
	cudaFree(resultSaveMeanOnGPU);
	cudaFree(resultSaveInvVarianceOnGPU);

	// -----------------------------
	// Indicate that main succeeded.
	// -----------------------------
	return 0;

} // main