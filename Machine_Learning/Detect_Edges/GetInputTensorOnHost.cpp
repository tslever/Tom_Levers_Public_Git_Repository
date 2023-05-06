
#include <vector>

#include <string>

#include <opencv2\opencv.hpp>


void GetInputTensorOnHost(
	double* inputTensor,
	double* pointerToTrainingImage,
	int images,
	int channels,
	int height,
	int width,
	std::string path,
	std::vector<std::string> filenames)
{

	cv::Size widthAndHeightOfImage = cv::Size(width, height);

	std::string pathToTrainingImage;
	int i;
	int h;
	int w;
	int c;
	cv::Mat trainingImage;

	for (i = 0; i < images; ++i)
	{
		pathToTrainingImage = path + filenames.at(i);

		trainingImage = cv::imread(pathToTrainingImage, cv::IMREAD_COLOR);

		cv::resize(
			trainingImage,
			trainingImage,
			widthAndHeightOfImage,
			0,
			0,
			cv::InterpolationFlags::INTER_LINEAR);

		trainingImage.convertTo(trainingImage, CV_64FC3);

		cv::normalize(trainingImage, trainingImage, 0, 1, cv::NORM_MINMAX);

		pointerToTrainingImage = trainingImage.ptr<double>(0);

		for (h = 0; h < height; ++h)
		{
			for (w = 0; w < width; ++w)
			{
				for (c = 0; c < channels; ++c)
				{
					inputTensor[w + h*width + c*height*width + i*channels*height*width] =
						pointerToTrainingImage[c + w*channels + h*width*channels];

				}
			}
		}
	}

}