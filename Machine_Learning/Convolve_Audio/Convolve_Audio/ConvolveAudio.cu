#include <stdio.h>
#include <stdint.h>
#include "ConvertIntDataToFloatData.h"
#include "PerformAugmentedConvolution.h"
#include "ConvertFloatDataToIntData.h"

int main() {

	// -----------------------------------------------------------------------------
	// Read "Hey_Ho_To_the_Greenwood--48_kHz.wav" and "Impulse_Response--48_kHz.wav"
	// into buffers pointed to by char (byte) pointers.
	// -----------------------------------------------------------------------------
	const char* filenameOfHeyHoToTheGreenwood = "Hey_Ho_To_The_Greenwood--48_kHz.wav";
	FILE* fileHeyHoToTheGreenwood = fopen(filenameOfHeyHoToTheGreenwood, "rb");
	fseek(fileHeyHoToTheGreenwood, 0, SEEK_END);
	long lengthOfHeyHoToTheGreenwood = ftell(fileHeyHoToTheGreenwood);
	long lengthOfCharDataOfHeyHoToTheGreenwood = lengthOfHeyHoToTheGreenwood - 44;
	fseek(fileHeyHoToTheGreenwood, 44, SEEK_SET);
	char* charDataOfHeyHoToTheGreenwood = new char[lengthOfCharDataOfHeyHoToTheGreenwood];
	fread(charDataOfHeyHoToTheGreenwood, sizeof(char), lengthOfCharDataOfHeyHoToTheGreenwood, fileHeyHoToTheGreenwood);
	fclose(fileHeyHoToTheGreenwood);
	fileHeyHoToTheGreenwood = NULL;

	const char* filenameOfImpulseResponse = "Impulse_Response--48_kHz.wav";
	FILE* fileImpulseResponse = fopen(filenameOfImpulseResponse, "rb");
	fseek(fileImpulseResponse, 0, SEEK_END);
	long lengthOfImpulseResponse = ftell(fileImpulseResponse);
	long lengthOfCharDataOfImpulseResponse = lengthOfImpulseResponse - 44;
	fseek(fileImpulseResponse, 44, SEEK_SET);
	char* charDataOfImpulseResponse = new char[lengthOfCharDataOfImpulseResponse];
	fread(charDataOfImpulseResponse, sizeof(char), lengthOfCharDataOfImpulseResponse, fileImpulseResponse);
	fclose(fileImpulseResponse);
	fileHeyHoToTheGreenwood = NULL;

	// -------------------------------------------------------------------------------
	// Convert char pointers to audio buffers to int16_t pointers to the same buffers.
	// -------------------------------------------------------------------------------
	int16_t* intDataOfHeyHoToTheGreenwood = (int16_t*)charDataOfHeyHoToTheGreenwood;
	int16_t* intDataOfImpulseResponse = (int16_t*)charDataOfImpulseResponse;

	// -----------------------------------------------------------------------------
	// Copy intDataOfHeyHoToTheGreenwood and intDataOfImpulseResponse into GPU heap.
	// -----------------------------------------------------------------------------
	long lengthOfIntDataOfHeyHoToTheGreenwood = lengthOfCharDataOfHeyHoToTheGreenwood / 2;
	long lengthOfIntDataOfImpulseResponse = lengthOfCharDataOfImpulseResponse / 2;

	int16_t* intDataOfHeyHoToTheGreenwood_GPU;
	cudaMalloc(&intDataOfHeyHoToTheGreenwood_GPU, lengthOfIntDataOfHeyHoToTheGreenwood * sizeof(int16_t));
	cudaMemcpy(intDataOfHeyHoToTheGreenwood_GPU, intDataOfHeyHoToTheGreenwood, lengthOfIntDataOfHeyHoToTheGreenwood * sizeof(int16_t), cudaMemcpyHostToDevice);

	int16_t* intDataOfImpulseResponse_GPU;
	cudaMalloc(&intDataOfImpulseResponse_GPU, lengthOfIntDataOfImpulseResponse * sizeof(int16_t));
	cudaMemcpy(intDataOfImpulseResponse_GPU, intDataOfImpulseResponse, lengthOfIntDataOfImpulseResponse * sizeof(int16_t), cudaMemcpyHostToDevice);

	// --------------------------------------------------------------------
	// Declare floatDataOfHeyToTheGreenwood and floatDataOfImpulseResponse.
	// --------------------------------------------------------------------
	long lengthOfFloatDataOfHeyHoToTheGreenwood = lengthOfIntDataOfHeyHoToTheGreenwood;
	float* floatDataOfHeyHoToTheGreenwood;
	cudaMalloc(&floatDataOfHeyHoToTheGreenwood, lengthOfFloatDataOfHeyHoToTheGreenwood * sizeof(float));

	long lengthOfFloatDataOfImpulseResponse = lengthOfIntDataOfImpulseResponse;
	float* floatDataOfImpulseResponse;
	cudaMalloc(&floatDataOfImpulseResponse, lengthOfFloatDataOfImpulseResponse * sizeof(float));

	// ----------------------------------------------------------------------------------------------------------------------------------
	// Convert intDataOfHeyHoToTheGreenwood to floatDataOfHeyHoToTheGreenwood and intDataOfImpulseResponse to floatDataOfImpulseResponse.
	// ----------------------------------------------------------------------------------------------------------------------------------
	int threadsPerBlock = 1024;
	long blocksForConvertingHeyHoToTheGreenwood = (lengthOfIntDataOfHeyHoToTheGreenwood + threadsPerBlock - 1) / threadsPerBlock;
	long blocksForConvertingImpulseResponse = (lengthOfIntDataOfImpulseResponse + threadsPerBlock - 1) / threadsPerBlock;

	ConvertIntDataToFloatData << <blocksForConvertingHeyHoToTheGreenwood, threadsPerBlock >> > (floatDataOfHeyHoToTheGreenwood, intDataOfHeyHoToTheGreenwood_GPU);
	cudaDeviceSynchronize();
	cudaFree(intDataOfHeyHoToTheGreenwood_GPU);

	ConvertIntDataToFloatData << <blocksForConvertingImpulseResponse, threadsPerBlock >> > (floatDataOfImpulseResponse, intDataOfImpulseResponse_GPU);
	cudaDeviceSynchronize();
	cudaFree(intDataOfHeyHoToTheGreenwood_GPU);

	// ------------------------------------------------------------------------------------------------
	// Perform augmented convolution of floatDataOfHeyHoToTheGreenwood with floatDataOfImpulseResponse.
	// ------------------------------------------------------------------------------------------------
	float* convolvedHeyHoToTheGreenwood;
	long lengthOfConvolvedHeyHoToTheGreenwood = lengthOfIntDataOfHeyHoToTheGreenwood - lengthOfIntDataOfImpulseResponse;
	cudaMalloc(&convolvedHeyHoToTheGreenwood, lengthOfConvolvedHeyHoToTheGreenwood * sizeof(float));

	long blocksForConvolving = (lengthOfConvolvedHeyHoToTheGreenwood + threadsPerBlock - 1) / threadsPerBlock;

	PerformAugmentedConvolution
		<< <blocksForConvolving, threadsPerBlock >> >
		(convolvedHeyHoToTheGreenwood, floatDataOfHeyHoToTheGreenwood, floatDataOfImpulseResponse, lengthOfFloatDataOfImpulseResponse);
	cudaDeviceSynchronize();
	cudaFree(floatDataOfHeyHoToTheGreenwood);
	cudaFree(floatDataOfImpulseResponse);

	// ----------------------------------------------------
	// Find maximumMagnitudeOfConvolvedHeyHoToTheGreenwood.
	// ----------------------------------------------------
	float* convolvedHeyHoToTheGreenwood_host = new float[lengthOfConvolvedHeyHoToTheGreenwood];
	cudaMemcpy(convolvedHeyHoToTheGreenwood_host, convolvedHeyHoToTheGreenwood, lengthOfConvolvedHeyHoToTheGreenwood * sizeof(float), cudaMemcpyDeviceToHost);

	float maximumMagnitudeOfConvolvedHeyHoToTheGreenwood = -32768.0; // = (-1.0) * pow(2.0, 15.0) - 1.0
	for (int i = 0; i < lengthOfConvolvedHeyHoToTheGreenwood; ++i) {
		if (convolvedHeyHoToTheGreenwood_host[i] > maximumMagnitudeOfConvolvedHeyHoToTheGreenwood) {
			maximumMagnitudeOfConvolvedHeyHoToTheGreenwood = convolvedHeyHoToTheGreenwood_host[i];
		}
	}
	cudaFree(convolvedHeyHoToTheGreenwood_host);

	// -----------------------------------------------------------------------------------------
	// Convert floatDataOfConvolvedHeyHoToTheGreenwood to intDataOfConvolvedHeyHoToTheGreenwood.
	// -----------------------------------------------------------------------------------------
	int16_t* intDataOfConvolvedHeyHoToTheGreenwood;
	cudaMalloc(&intDataOfConvolvedHeyHoToTheGreenwood, lengthOfConvolvedHeyHoToTheGreenwood * sizeof(int16_t));

	long blocksForConvertingConvolvedHeyHoToTheGreenwood = blocksForConvolving;

	ConvertFloatDataToIntData
		<< <blocksForConvertingConvolvedHeyHoToTheGreenwood, threadsPerBlock >> >
		(intDataOfConvolvedHeyHoToTheGreenwood, convolvedHeyHoToTheGreenwood, maximumMagnitudeOfConvolvedHeyHoToTheGreenwood);
	cudaDeviceSynchronize();
	cudaFree(convolvedHeyHoToTheGreenwood);

	// ------------------------------------------------------------------------------------------------
	// Write intDataOfConvolvedHeyHoToTheGreenwood to "Convolved_Hey_Ho_To_the_Greenwood--48_kHz.wav".
	// ------------------------------------------------------------------------------------------------
	// Copy intDataOfConvolvedHeyHoToTheGreenwood to host heap.
	int16_t* intDataOfConvolvedHeyHoToTheGreenwood_host = new int16_t[lengthOfConvolvedHeyHoToTheGreenwood];
	cudaMemcpy(intDataOfConvolvedHeyHoToTheGreenwood_host, intDataOfConvolvedHeyHoToTheGreenwood, lengthOfConvolvedHeyHoToTheGreenwood * sizeof(int16_t), cudaMemcpyDeviceToHost);
	cudaFree(intDataOfConvolvedHeyHoToTheGreenwood);

	// Open fileConvolvedHeyHoToTheGreenwood for writing.
	FILE* fileConvolvedHeyHoToTheGreenwood = fopen("Hey_Ho_To_The_Greenwood--Convolved--48_kHz.wav", "wb");

	// Write RIFF header.
	unsigned char chunkID[4] = { 'R', 'I', 'F', 'F' };
	fwrite(&chunkID, sizeof(char), 4, fileConvolvedHeyHoToTheGreenwood);

	uint32_t chunkSize = lengthOfHeyHoToTheGreenwood - 8;
	fwrite(&chunkSize, sizeof(uint32_t), 1, fileConvolvedHeyHoToTheGreenwood);

	unsigned char format[4] = { 'W', 'A', 'V', 'E' };
	fwrite(&format, sizeof(char), 4, fileConvolvedHeyHoToTheGreenwood);

	// Write format subchunk.
	unsigned char subchunk1ID[4] = { 'f', 'm', 't', ' ' };
	fwrite(&subchunk1ID, sizeof(char), 4, fileConvolvedHeyHoToTheGreenwood);

	uint32_t subchunk1Size = 16; // for PCM
	fwrite(&subchunk1Size, sizeof(uint32_t), 1, fileConvolvedHeyHoToTheGreenwood);

	uint16_t audioFormat = 1; // for PCM
	fwrite(&audioFormat, sizeof(uint16_t), 1, fileConvolvedHeyHoToTheGreenwood);

	uint16_t numChannels = 1; // for PCM
	fwrite(&numChannels, sizeof(uint16_t), 1, fileConvolvedHeyHoToTheGreenwood);

	uint32_t sampleRate = 48000;
	fwrite(&sampleRate, sizeof(uint32_t), 1, fileConvolvedHeyHoToTheGreenwood);

	uint16_t bitsPerSample = 16;

	uint32_t byteRate = 96000; // = (uint32_t)numChannels * sampleRate * (uint32_t)bitsPerSample/8;
	fwrite(&byteRate, sizeof(uint32_t), 1, fileConvolvedHeyHoToTheGreenwood);

	uint16_t blockAlign = 2; // = (uint32_t)numChannels * (uint32_t)bitsPerSample/8;
	fwrite(&blockAlign, sizeof(uint16_t), 1, fileConvolvedHeyHoToTheGreenwood);

	fwrite(&bitsPerSample, sizeof(uint16_t), 1, fileConvolvedHeyHoToTheGreenwood);

	// --------------------
	// Write data subchunk.
	// --------------------
	unsigned char subchunk2ID[4] = { 'd', 'a', 't', 'a' };
	fwrite(&subchunk2ID, sizeof(char), 4, fileConvolvedHeyHoToTheGreenwood);

	uint32_t numSamples = lengthOfConvolvedHeyHoToTheGreenwood;

	uint32_t subchunk2Size = numSamples * (2 /*= (uint32_t)numChannels * (uint32_t)bitsPerSample/8*/);
	fwrite(&subchunk2Size, sizeof(uint32_t), 1, fileConvolvedHeyHoToTheGreenwood);

	fwrite(intDataOfConvolvedHeyHoToTheGreenwood_host, sizeof(int16_t), numSamples, fileConvolvedHeyHoToTheGreenwood);
	delete intDataOfConvolvedHeyHoToTheGreenwood_host;

	fclose(fileConvolvedHeyHoToTheGreenwood);
	fileConvolvedHeyHoToTheGreenwood = NULL;

	// ---------
	// Clean up.
	// ---------
	delete charDataOfHeyHoToTheGreenwood;

	// -----------------------------------------
	// Indicate main has completed successfully.
	// -----------------------------------------
	return 0;

} // main