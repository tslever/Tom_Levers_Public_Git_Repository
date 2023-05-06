#ifndef IM2COL_H
#define IM2COL_H


float im2col_get_pixel(float* im, int height, int width, int channels,
    int row, int col, int channel, int pad);
void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col);
void im2col_cpu_ext(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col);


#endif