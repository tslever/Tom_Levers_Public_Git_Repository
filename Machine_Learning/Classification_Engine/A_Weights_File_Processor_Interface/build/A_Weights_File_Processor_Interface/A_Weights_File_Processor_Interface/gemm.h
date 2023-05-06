#ifndef GEMM_H
#define GEMM_H


#include <stdint.h>
#include "darknet.h"


int is_avx();
void repack_input(float* input, float* re_packed_input, int w, int h, int c);
void float_to_bit(float* src, unsigned char* dst, size_t size);
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col);
int is_fma_avx2();
void transpose_uint32(uint32_t* src, uint32_t* dst, int src_h, int src_w, int src_align, int dst_align);
void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char* A, int lda,
    unsigned char* B, int ldb,
    float* C, int ldc, float* mean_arr);
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align);
static inline void set_bit(unsigned char* const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
    //dst[dst_i] |= 1 << (8 - dst_shift);
}
void transpose_bin(uint32_t* A, uint32_t* B, const int n, const int m,
    const int lda, const int ldb, const int block_size);
void activate_array_cpu_custom(float* x, const int n, const ACTIVATION a);
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float BETA,
    float* C, int ldc);
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float BETA,
    float* C, int ldc);
void transpose_32x32_bits_reversed_diagonale(uint32_t* A, uint32_t* B, int m, int n);
static inline unsigned char get_bit(unsigned char const* const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    //unsigned char val = (src[src_i] & (1 << (8 - src_shift))) > 0;
    return val;
}
void forward_maxpool_layer_avx(float* src, float* dst, int* indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch);


#endif