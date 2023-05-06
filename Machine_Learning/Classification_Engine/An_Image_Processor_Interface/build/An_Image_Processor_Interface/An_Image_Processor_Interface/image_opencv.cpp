#include "pch.h"
#include "darknet.h"
#include <opencv2/opencv.hpp>
#include "image_opencv.h"
#include <fstream>


cv::Mat load_image_mat(char* filename, int channels)
{
    int flag = cv::IMREAD_UNCHANGED;
    if (channels == 0) flag = cv::IMREAD_COLOR;
    else if (channels == 1) flag = cv::IMREAD_GRAYSCALE;
    else if (channels == 3) flag = cv::IMREAD_COLOR;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    //flag |= IMREAD_IGNORE_ORIENTATION;    // un-comment it if you want

    cv::Mat* mat_ptr = (cv::Mat*)load_image_mat_cv(filename, flag);

    if (mat_ptr == NULL) {
        return cv::Mat();
    }
    cv::Mat mat = *mat_ptr;
    delete mat_ptr;

    return mat;
}


mat_cv* load_image_mat_cv(const char* filename, int flag)
{
    cv::Mat* mat_ptr = NULL;
    try {
        cv::Mat mat = cv::imread(filename, flag);
        if (mat.empty())
        {
            std::string shrinked_filename = filename;
            if (shrinked_filename.length() > 1024) {
                shrinked_filename.resize(1024);
                shrinked_filename = std::string("name is too long: ") + shrinked_filename;
            }
            std::cerr << "Cannot load image " << shrinked_filename << std::endl;
            std::ofstream bad_list("bad.list", std::ios::out | std::ios::app);
            bad_list << shrinked_filename << std::endl;
            //if (check_mistakes) getchar();
            return NULL;
        }
        cv::Mat dst;
        if (mat.channels() == 3) cv::cvtColor(mat, dst, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, dst, cv::COLOR_RGBA2BGRA);
        else dst = mat;

        mat_ptr = new cv::Mat(dst);

        return (mat_cv*)mat_ptr;
    }
    catch (...) {
        std::cerr << "OpenCV exception: load_image_mat_cv \n";
    }
    if (mat_ptr) delete mat_ptr;
    return NULL;
}


image load_image_cv(char* filename, int channels)
{
    cv::Mat mat = load_image_mat(filename, channels);

    if (mat.empty()) {
        return make_image(10, 10, channels);
    }
    return mat_to_image(mat);
}


image mat_to_image(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char* data = (unsigned char*)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
                //im.data[k*w*h + y*w + x] = val / 255.0f;

                im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
            }
        }
    }
    return im;
}