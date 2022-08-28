
#include <opencv2\opencv.hpp>


void DisplayImagesInActivationOutputTensor(
    double* activationOutputTensor,
    int subtensors,
    int height,
    int width,
    int channels)
{
    int elementsInOutputImage = height * width * channels;

    for (int i = 0; i < subtensors; ++i)
    {
        double* pointerToOutputImage = (double*)malloc(elementsInOutputImage * sizeof(double));

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < channels; ++c)
                {
                    pointerToOutputImage[c + w*channels + h*width*channels] =
                        activationOutputTensor[w + h*width + c*height*width + i*channels*height*width];
                }
            }
        }

        cv::Mat outputImage(height, width, CV_64FC3, pointerToOutputImage);

        cv::normalize(outputImage, outputImage, 0.0, 255.0, cv::NORM_MINMAX);

        outputImage.convertTo(outputImage, CV_8UC3);

        cv::namedWindow("Display Window");

        cv::imshow("Display Window", outputImage);

        cv::waitKey(0);

        delete pointerToOutputImage;
    }
}