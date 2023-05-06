#include "pch.h"
#include "gemm.h"
#include <vcruntime_string.h>
#include <immintrin.h>
#include <stdint.h>
#include "im2col.h"
#include <intrin.h>
#include <stdio.h>
#include "activations.h"
#include <float.h>


#define TILE_M 4 // 4 ops
#define TILE_N 16 // AVX2 = 2 ops * 8 floats
#define TILE_K 16 // loop
#define PUT_IN_REGISTER


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input(float* input, float* re_packed_input, int w, int h, int c)
{
    const int items_per_channel = w * h;
    int chan, i;
    for (chan = 0; chan < c; chan += 32)
    {
        for (i = 0; i < items_per_channel; ++i)
        {
            int c_pack;
            for (c_pack = 0; c_pack < 32; ++c_pack) {
                float src = input[(chan + c_pack) * items_per_channel + i];

                re_packed_input[chan * items_per_channel + i * 32 + c_pack] = src;
            }
        }
    }
}


void float_to_bit(float* src, unsigned char* dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    //__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
    __m256 float_zero256 = _mm256_set1_ps(0.0);

    for (i = 0; i < size; i += 8)
    {
        //__m256i src256 = _mm256_loadu_si256((__m256i *)(&src[i]));
        //__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
        //uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
        ////mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

        __m256 src256 = _mm256_loadu_ps((float*)(&src[i]));
        __m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
        uint32_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

        dst[i / 8] = mask;
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
    {
#pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((float*)(&data_im[im_col + width * (im_row + height * c_im)]));
                    _mm256_storeu_ps(&data_col[col_index], src256);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = data_im[im_col + width * (im_row + height * c_im)];
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        //printf("\n Error: is no non-optimized version \n");
        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    }
}


#define cpuid(info, x)    __cpuidex(info, x, 0)
//  SIMD: 256-bit
static int HW_AVX, HW_XOP, HW_FMA3, HW_FMA4, HW_AVX2;
// Misc.
static int HW_MMX, HW_x64, HW_RDRAND, HW_BMI1, HW_BMI2, HW_ADX, HW_PREFETCHWT1;
static int HW_ABM;      // Advanced Bit Manipulation
//  SIMD: 128-bit
static int HW_SSE, HW_SSE2, HW_SSE3, HW_SSSE3, HW_SSE41, HW_SSE42, HW_SSE4a, HW_AES, HW_SHA;
//  SIMD: 512-bit
static int HW_AVX512F;    //  AVX512 Foundation
static int HW_AVX512CD;   //  AVX512 Conflict Detection
static int HW_AVX512PF;   //  AVX512 Prefetch
static int HW_AVX512ER;   //  AVX512 Exponential + Reciprocal
static int HW_AVX512VL;   //  AVX512 Vector Length Extensions
static int HW_AVX512BW;   //  AVX512 Byte + Word
static int HW_AVX512DQ;   //  AVX512 Doubleword + Quadword
static int HW_AVX512IFMA; //  AVX512 Integer 52-bit Fused Multiply-Add
static int HW_AVX512VBMI; //  AVX512 Vector Byte Manipulation Instructions


// https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set
void check_cpu_features(void) {
    int info[4];
    cpuid(info, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000);
    unsigned nExIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001) {
        cpuid(info, 0x00000001);
        HW_MMX = (info[3] & ((uint32_t)1 << 23)) != 0;
        HW_SSE = (info[3] & ((uint32_t)1 << 25)) != 0;
        HW_SSE2 = (info[3] & ((uint32_t)1 << 26)) != 0;
        HW_SSE3 = (info[2] & ((uint32_t)1 << 0)) != 0;

        HW_SSSE3 = (info[2] & ((uint32_t)1 << 9)) != 0;
        HW_SSE41 = (info[2] & ((uint32_t)1 << 19)) != 0;
        HW_SSE42 = (info[2] & ((uint32_t)1 << 20)) != 0;
        HW_AES = (info[2] & ((uint32_t)1 << 25)) != 0;

        HW_AVX = (info[2] & ((uint32_t)1 << 28)) != 0;
        HW_FMA3 = (info[2] & ((uint32_t)1 << 12)) != 0;

        HW_RDRAND = (info[2] & ((uint32_t)1 << 30)) != 0;
    }
    if (nIds >= 0x00000007) {
        cpuid(info, 0x00000007);
        HW_AVX2 = (info[1] & ((uint32_t)1 << 5)) != 0;

        HW_BMI1 = (info[1] & ((uint32_t)1 << 3)) != 0;
        HW_BMI2 = (info[1] & ((uint32_t)1 << 8)) != 0;
        HW_ADX = (info[1] & ((uint32_t)1 << 19)) != 0;
        HW_SHA = (info[1] & ((uint32_t)1 << 29)) != 0;
        HW_PREFETCHWT1 = (info[2] & ((uint32_t)1 << 0)) != 0;

        HW_AVX512F = (info[1] & ((uint32_t)1 << 16)) != 0;
        HW_AVX512CD = (info[1] & ((uint32_t)1 << 28)) != 0;
        HW_AVX512PF = (info[1] & ((uint32_t)1 << 26)) != 0;
        HW_AVX512ER = (info[1] & ((uint32_t)1 << 27)) != 0;
        HW_AVX512VL = (info[1] & ((uint32_t)1 << 31)) != 0;
        HW_AVX512BW = (info[1] & ((uint32_t)1 << 30)) != 0;
        HW_AVX512DQ = (info[1] & ((uint32_t)1 << 17)) != 0;
        HW_AVX512IFMA = (info[1] & ((uint32_t)1 << 21)) != 0;
        HW_AVX512VBMI = (info[2] & ((uint32_t)1 << 1)) != 0;
    }
    if (nExIds >= 0x80000001) {
        cpuid(info, 0x80000001);
        HW_x64 = (info[3] & ((uint32_t)1 << 29)) != 0;
        HW_ABM = (info[2] & ((uint32_t)1 << 5)) != 0;
        HW_SSE4a = (info[2] & ((uint32_t)1 << 6)) != 0;
        HW_FMA4 = (info[2] & ((uint32_t)1 << 16)) != 0;
        HW_XOP = (info[2] & ((uint32_t)1 << 11)) != 0;
    }
}


int is_fma_avx2() {
    static int result = -1;
    if (result == -1) {
        check_cpu_features();
        result = HW_FMA3 && HW_AVX2;
        if (result == 1) printf(" Used FMA & AVX2 \n");
        else printf(" Not used FMA & AVX2 \n");
    }
    return result;
}


void transpose_uint32(uint32_t* src, uint32_t* dst, int src_h, int src_w, int src_align, int dst_align)
{
    //l.bit_align - algined (n) by 32
    //new_ldb - aligned (k) by 256

    int i;
    //#pragma omp parallel for
    for (i = 0; i < src_h; i += 1)  // l.size*l.size*l.c;
    {
        int j;
        for (j = 0; j < src_w; j += 1)  // out_h*out_w;
        {
            ((uint32_t*)dst)[j * dst_align / 32 + i] = ((uint32_t*)src)[i * src_align + j];
        }
    }
}


static inline __m256i count256(__m256i v) {
    __m256i lookup =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
            2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
            1, 2, 2, 3, 2, 3, 3, 4);

    __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i total = _mm256_add_epi8(popcnt1, popcnt2);

    return _mm256_sad_epu8(total, _mm256_setzero_si256());
}


static inline void xnor_avx2_popcnt(__m256i a_bit256, __m256i b_bit256, __m256i* count_sum) {
    __m256i c_bit256 = _mm256_set1_epi8((char)255);

    __m256i xor256 = _mm256_xor_si256(a_bit256, b_bit256);  // xnor = not(xor(a,b))
    c_bit256 = _mm256_andnot_si256(xor256, c_bit256);  // can be optimized - we can do other NOT for wegihts once and do not do this NOT

    *count_sum = _mm256_add_epi64(count256(c_bit256), *count_sum);    //  1st part - popcnt Mula's algorithm
}


// 2nd part - popcnt Mula's algorithm
static inline int get_count_mula(__m256i count_sum) {
    return _mm256_extract_epi64(count_sum, 0)
        + _mm256_extract_epi64(count_sum, 1)
        + _mm256_extract_epi64(count_sum, 2)
        + _mm256_extract_epi64(count_sum, 3);
}


// 5x times faster than gemm()-float32
// further optimizations: do mean-mult only for the last layer
void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char* A, int lda,
    unsigned char* B, int ldb,
    float* C, int ldc, float* mean_arr)
{
    int i;

#if defined(_OPENMP)
    static int max_num_threads = 0;
    if (max_num_threads == 0) {
        max_num_threads = omp_get_max_threads();
        //omp_set_num_threads(max_num_threads / 2);
    }
#endif

    //#pragma omp parallel for
    //for (i = 0; i < M; ++i)
#pragma omp parallel for
    for (i = 0; i < (M / 2) * 2; i += 2)
    {   // l.n - filters [16 - 55 - 1024]
        float mean_val_0 = mean_arr[i + 0];
        float mean_val_1 = mean_arr[i + 1];
        int j, k;
        //__m256i all_1 = _mm256_set1_epi8(255);

        //for (j = 0; j < N; ++j)
        for (j = 0; j < (N / 2) * 2; j += 2)
        { // out_h*out_w - one channel output size [169 - 173056]
            //int count = 0;
            const int bit_step = 256;
            __m256i count_sum_0 = _mm256_set1_epi8(0);
            __m256i count_sum_1 = _mm256_set1_epi8(0);
            __m256i count_sum_2 = _mm256_set1_epi8(0);
            __m256i count_sum_3 = _mm256_set1_epi8(0);

            for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                __m256i a_bit256_0 = _mm256_loadu_si256((__m256i*)(A + ((i + 0) * lda + k) / 8));
                __m256i b_bit256_0 = _mm256_loadu_si256((__m256i*)(B + ((j + 0) * ldb + k) / 8));

                __m256i a_bit256_1 = _mm256_loadu_si256((__m256i*)(A + ((i + 1) * lda + k) / 8));
                __m256i b_bit256_1 = _mm256_loadu_si256((__m256i*)(B + ((j + 1) * ldb + k) / 8));


                xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum_0);
                xnor_avx2_popcnt(a_bit256_0, b_bit256_1, &count_sum_1);

                xnor_avx2_popcnt(a_bit256_1, b_bit256_0, &count_sum_2);
                xnor_avx2_popcnt(a_bit256_1, b_bit256_1, &count_sum_3);

                //count += popcnt256(c_bit256);
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            int count_0 = get_count_mula(count_sum_0);
            int count_1 = get_count_mula(count_sum_1);
            int count_2 = get_count_mula(count_sum_2);
            int count_3 = get_count_mula(count_sum_3);

            const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count_0 = count_0 - f1;    // remove extra bits (from empty space for align only)
            count_1 = count_1 - f1;
            count_2 = count_2 - f1;
            count_3 = count_3 - f1;
            C[i * ldc + (j + 0)] = (2 * count_0 - K) * mean_val_0;
            C[i * ldc + (j + 1)] = (2 * count_1 - K) * mean_val_0;
            C[(i + 1) * ldc + (j + 0)] = (2 * count_2 - K) * mean_val_1;
            C[(i + 1) * ldc + (j + 1)] = (2 * count_3 - K) * mean_val_1;
        }

        int i_d;
        for (i_d = 0; i_d < 2; ++i_d)
        {
            float mean_val = mean_arr[i + i_d];
            for (j = (N / 2) * 2; j < N; j += 1)
            { // out_h*out_w - one channel output size [169 - 173056]
                const int bit_step = 256;
                __m256i count_sum = _mm256_set1_epi8(0);

                for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                    __m256i a_bit256_0 = _mm256_loadu_si256((__m256i*)(A + ((i + i_d + 0) * lda + k) / 8));
                    __m256i b_bit256_0 = _mm256_loadu_si256((__m256i*)(B + ((j + 0) * ldb + k) / 8));
                    xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum);
                }
                int count = get_count_mula(count_sum);
                const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits (from empty space for align only)
                C[(i + i_d) * ldc + j] = (2 * count - K) * mean_val;
            }
        }
    }

    for (i = (M / 2) * 2; i < M; i += 1)
    {
        float mean_val = mean_arr[i];
        int j, k;
        for (j = 0; j < N; j += 1)
        { // out_h*out_w - one channel output size [169 - 173056]
            const int bit_step = 256;
            __m256i count_sum = _mm256_set1_epi8(0);

            for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                __m256i a_bit256_0 = _mm256_loadu_si256((__m256i*)(A + ((i + 0) * lda + k) / 8));
                __m256i b_bit256_0 = _mm256_loadu_si256((__m256i*)(B + ((j + 0) * ldb + k) / 8));
                xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum);
            }
            int count = get_count_mula(count_sum);
            const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count = count - f1;    // remove extra bits (from empty space for align only)
            C[i * ldc + j] = (2 * count - K) * mean_val;
        }
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
    {
        __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
        __m256 float_zero256 = _mm256_set1_ps(0.00);

        int new_ldb = bit_align;

#pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //__m256i src256 = _mm256_loadu_si256((__m256i *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    //__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
                    //uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
                    //mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

                    __m256 src256 = _mm256_loadu_ps((float*)(&data_im[im_col + width * (im_row + height * c_im)]));
                    __m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
                    uint16_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

                    uint16_t* dst_ptr = (uint16_t*)&((uint8_t*)data_col)[col_index / 8];
                    *dst_ptr |= (mask << (col_index % 8));
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    float val = data_im[im_col + width * (im_row + height * c_im)];
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}


void activate_array_cpu_custom(float* x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        if (is_fma_avx2()) {
            __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
            __m256 all256_01 = _mm256_set1_ps(0.1F);

            for (i = 0; i < n - 8; i += 8) {
                //x[i] = (x[i]>0) ? x[i] : .1*x[i];

                __m256 src256 = _mm256_loadu_ps(&x[i]);
                __m256 mult256 = _mm256_mul_ps((src256), all256_01); // mult * 0.1

                __m256i sign256 = _mm256_and_si256(_mm256_castps_si256(src256), all256_sing1); // check sign in 8 x 32-bit floats

                __m256 result256 = _mm256_blendv_ps(src256, mult256, _mm256_castsi256_ps(sign256)); // (sign>0) ? src : mult;
                _mm256_storeu_ps(&x[i], result256);
            }
        }

        for (; i < n; ++i) {
            x[i] = (x[i] > 0) ? x[i] : .1 * x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float BETA,
    float* C, int ldc)
{
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}


void gemm_nn_fast(int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    int i;

#pragma omp parallel for
    for (i = 0; i < (M / TILE_M) * TILE_M; i += TILE_M)
    {
        int j, k;
        int i_d, k_d;

        for (k = 0; k < (K / TILE_K) * TILE_K; k += TILE_K)
        {
            for (j = 0; j < (N / TILE_N) * TILE_N; j += TILE_N)
            {
                // L1 - 6 bits tag [11:6] - cache size 32 KB, conflict for each 4 KB
                // L2 - 9 bits tag [14:6] - cache size 256 KB, conflict for each 32 KB
                // L3 - 13 bits tag [18:6] - cache size 8 MB, conflict for each 512 KB

                __m256 result256;
                __m256 a256_0, b256_0;    // AVX
                __m256 a256_1, b256_1;    // AVX
                __m256 a256_2;// , b256_2;    // AVX
                __m256 a256_3;// , b256_3;    // AVX
                __m256 c256_0, c256_1, c256_2, c256_3;
                __m256 c256_4, c256_5, c256_6, c256_7;

                c256_0 = _mm256_loadu_ps(&C[(0 + i) * ldc + (0 + j)]);
                c256_1 = _mm256_loadu_ps(&C[(1 + i) * ldc + (0 + j)]);
                c256_2 = _mm256_loadu_ps(&C[(0 + i) * ldc + (8 + j)]);
                c256_3 = _mm256_loadu_ps(&C[(1 + i) * ldc + (8 + j)]);

                c256_4 = _mm256_loadu_ps(&C[(2 + i) * ldc + (0 + j)]);
                c256_5 = _mm256_loadu_ps(&C[(3 + i) * ldc + (0 + j)]);
                c256_6 = _mm256_loadu_ps(&C[(2 + i) * ldc + (8 + j)]);
                c256_7 = _mm256_loadu_ps(&C[(3 + i) * ldc + (8 + j)]);


                for (k_d = 0; k_d < (TILE_K); ++k_d)
                {
                    a256_0 = _mm256_set1_ps(ALPHA * A[(0 + i) * lda + (k_d + k)]);
                    a256_1 = _mm256_set1_ps(ALPHA * A[(1 + i) * lda + (k_d + k)]);

                    a256_2 = _mm256_set1_ps(ALPHA * A[(2 + i) * lda + (k_d + k)]);
                    a256_3 = _mm256_set1_ps(ALPHA * A[(3 + i) * lda + (k_d + k)]);


                    b256_0 = _mm256_loadu_ps(&B[(k_d + k) * ldb + (0 + j)]);
                    b256_1 = _mm256_loadu_ps(&B[(k_d + k) * ldb + (8 + j)]);

                    // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                    //c256_0 = _mm256_fmadd_ps(a256_0, b256_0, c256_0);
                    //c256_1 = _mm256_fmadd_ps(a256_1, b256_0, c256_1);
                    //c256_2 = _mm256_fmadd_ps(a256_0, b256_1, c256_2);
                    //c256_3 = _mm256_fmadd_ps(a256_1, b256_1, c256_3);

                    //c256_4 = _mm256_fmadd_ps(a256_2, b256_0, c256_4);
                    //c256_5 = _mm256_fmadd_ps(a256_3, b256_0, c256_5);
                    //c256_6 = _mm256_fmadd_ps(a256_2, b256_1, c256_6);
                    //c256_7 = _mm256_fmadd_ps(a256_3, b256_1, c256_7);

                    result256 = _mm256_mul_ps(a256_0, b256_0);
                    c256_0 = _mm256_add_ps(result256, c256_0);

                    result256 = _mm256_mul_ps(a256_1, b256_0);
                    c256_1 = _mm256_add_ps(result256, c256_1);

                    result256 = _mm256_mul_ps(a256_0, b256_1);
                    c256_2 = _mm256_add_ps(result256, c256_2);

                    result256 = _mm256_mul_ps(a256_1, b256_1);
                    c256_3 = _mm256_add_ps(result256, c256_3);


                    result256 = _mm256_mul_ps(a256_2, b256_0);
                    c256_4 = _mm256_add_ps(result256, c256_4);

                    result256 = _mm256_mul_ps(a256_3, b256_0);
                    c256_5 = _mm256_add_ps(result256, c256_5);

                    result256 = _mm256_mul_ps(a256_2, b256_1);
                    c256_6 = _mm256_add_ps(result256, c256_6);

                    result256 = _mm256_mul_ps(a256_3, b256_1);
                    c256_7 = _mm256_add_ps(result256, c256_7);
                }
                _mm256_storeu_ps(&C[(0 + i) * ldc + (0 + j)], c256_0);
                _mm256_storeu_ps(&C[(1 + i) * ldc + (0 + j)], c256_1);
                _mm256_storeu_ps(&C[(0 + i) * ldc + (8 + j)], c256_2);
                _mm256_storeu_ps(&C[(1 + i) * ldc + (8 + j)], c256_3);

                _mm256_storeu_ps(&C[(2 + i) * ldc + (0 + j)], c256_4);
                _mm256_storeu_ps(&C[(3 + i) * ldc + (0 + j)], c256_5);
                _mm256_storeu_ps(&C[(2 + i) * ldc + (8 + j)], c256_6);
                _mm256_storeu_ps(&C[(3 + i) * ldc + (8 + j)], c256_7);
            }

            for (j = (N / TILE_N) * TILE_N; j < N; ++j) {
                for (i_d = i; i_d < (i + TILE_M); ++i_d)
                {
                    for (k_d = k; k_d < (k + TILE_K); ++k_d)
                    {
                        PUT_IN_REGISTER float A_PART = ALPHA * A[i_d * lda + k_d];
                        C[i_d * ldc + j] += A_PART * B[k_d * ldb + j];
                    }
                }
            }
        }

        for (k = (K / TILE_K) * TILE_K; k < K; ++k)
        {
            for (i_d = i; i_d < (i + TILE_M); ++i_d)
            {
                PUT_IN_REGISTER float A_PART = ALPHA * A[i_d * lda + k];
                for (j = 0; j < N; ++j) {
                    C[i_d * ldc + j] += A_PART * B[k * ldb + j];
                }
            }
        }
    }

    for (i = (M / TILE_M) * TILE_M; i < M; ++i) {
        int j, k;
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}


// https://software.intel.com/sites/landingpage/IntrinsicsGuide
void gemm_nn(int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    int i, j, k;
    if (is_avx() == 1) {    // AVX
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                float A_PART = ALPHA * A[i * lda + k];
                __m256 a256, b256, c256, result256;    // AVX
                a256 = _mm256_set1_ps(A_PART);
                for (j = 0; j < N - 8; j += 8) {
                    b256 = _mm256_loadu_ps(&B[k * ldb + j]);
                    c256 = _mm256_loadu_ps(&C[i * ldc + j]);
                    // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                    //result256 = _mm256_fmadd_ps(a256, b256, c256);
                    result256 = _mm256_mul_ps(a256, b256);
                    result256 = _mm256_add_ps(result256, c256);
                    _mm256_storeu_ps(&C[i * ldc + j], result256);
                }

                int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;
                for (j = prev_end; j < N; ++j)
                    C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
    else {
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
                for (j = 0; j < N; ++j) {
                    C[i * ldc + j] += A_PART * B[k * ldb + j];
                }
                /* // SSE
                __m128 a128, b128, c128, result128;    // SSE
                a128 = _mm_set1_ps(A_PART);
                for (j = 0; j < N - 4; j += 4) {
                b128 = _mm_loadu_ps(&B[k*ldb + j]);
                c128 = _mm_loadu_ps(&C[i*ldc + j]);
                //result128 = _mm_fmadd_ps(a128, b128, c128);
                result128 = _mm_mul_ps(a128, b128);
                result128 = _mm_add_ps(result128, c128);
                _mm_storeu_ps(&C[i*ldc + j], result128);
                }

                int prev_end = (N % 4 == 0) ? (N - 4) : (N / 4) * 4;
                for (j = prev_end; j < N; ++j){
                C[i*ldc + j] += A_PART*B[k*ldb + j];
                }
                */
            }
        }
    }
}


void gemm_tn(int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}


void gemm_nt(int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            PUT_IN_REGISTER float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}


void gemm_tt(int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            PUT_IN_REGISTER float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float BETA,
    float* C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    if (BETA != 1) {
        int i, j;
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] *= BETA;
            }
        }
    }

    is_avx();   // initialize static variable
    if (is_fma_avx2() && !TA && !TB) {
        gemm_nn_fast(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else {
        int t;
#pragma omp parallel for
        for (t = 0; t < M; ++t) {
            if (!TA && !TB)
                gemm_nn(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
            else if (TA && !TB)
                gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
            else if (!TA && TB)
                gemm_nt(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
            else
                gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
        }
    }
}


int is_avx() {
    static int result = -1;
    if (result == -1) {
        check_cpu_features();
        result = HW_AVX;
        if (result == 1) printf(" Used AVX \n");
        else printf(" Not used AVX \n");
    }
    return result;
}





// transpose by 32-bit
void transpose_bin(uint32_t* A, uint32_t* B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    //printf("\n n = %d (n mod 32 = %d), m = %d (m mod 32 = %d) \n", n, n % 32, m, m % 32);
    //printf("\n lda = %d (lda mod 32 = %d), ldb = %d (ldb mod 32 = %d) \n", lda, lda % 32, ldb, ldb % 32);
    int i;
#pragma omp parallel for
    for (i = 0; i < n; i += 32) {
        int j;
        for (j = 0; j < m; j += 32) {
            int a_index = i * lda + j;
            int b_index = j * ldb + i;
            transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
            //transpose_32x32_bits_my(&A[a_index/32], &B[b_index/32], lda/32, ldb/32);
        }
        for (; j < m; ++j) {
            if (get_bit((const unsigned char* const)A, i * lda + j)) set_bit((unsigned char* const)B, j * ldb + i);
        }
    }
}


#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);


uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}


uint32_t reverse_32_bit(uint32_t a)
{
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}


void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    //m = 0x0000FFFF;
    //for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
    //    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
    //        t = (A[k] ^ (A[k + j] >> j)) & m;
    //        A[k] = A[k] ^ t;
    //        A[k + j] = A[k + j] ^ (t << j);
    //    }
    //}

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}


void transpose_32x32_bits_reversed_diagonale(uint32_t* A, uint32_t* B, int m, int n)
{
    unsigned A_tmp[32];
    int i;
#pragma unroll
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
#pragma unroll
    for (i = 0; i < 32; ++i) B[i * n] = A_tmp[i];
}


void forward_maxpool_layer_avx(float* src, float* dst, int* indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{

    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;
    int b, k;

    for (b = 0; b < batch; ++b) {
#pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                //for (j = 0; j < out_w; ++j) {
                j = 0;

                if (stride == 1 && is_avx() == 1) {
                    for (j = 0; j < out_w - 8 - (size - 1); j += 8) {
                        int out_index = j + out_w * (i + out_h * (k + c * b));
                        __m256 max256 = _mm256_set1_ps(-FLT_MAX);
                        for (n = 0; n < size; ++n) {
                            for (m = 0; m < size; ++m) {
                                int cur_h = h_offset + i * stride + n;
                                int cur_w = w_offset + j * stride + m;
                                int index = cur_w + w * (cur_h + h * (k + b * c));
                                int valid = (cur_h >= 0 && cur_h < h&&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                max256 = _mm256_max_ps(src256, max256);
                            }
                        }
                        _mm256_storeu_ps(&dst[out_index], max256);

                    }
                }
                else if (size == 2 && stride == 2 && is_avx() == 1) {
                    for (j = 0; j < out_w - 4; j += 4) {
                        int out_index = j + out_w * (i + out_h * (k + c * b));
                        //float max = -FLT_MAX;
                        //int max_i = -1;
                        __m128 max128 = _mm_set1_ps(-FLT_MAX);

                        for (n = 0; n < size; ++n) {
                            //for (m = 0; m < size; ++m)
                            m = 0;
                            {
                                int cur_h = h_offset + i * stride + n;
                                int cur_w = w_offset + j * stride + m;
                                int index = cur_w + w * (cur_h + h * (k + b * c));
                                int valid = (cur_h >= 0 && cur_h < h&&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                __m256 src256_2 = _mm256_permute_ps(src256, (1 << 0) | (3 << 4));
                                __m256 max256 = _mm256_max_ps(src256, src256_2);

                                __m128 src128_0 = _mm256_extractf128_ps(max256, 0);
                                __m128 src128_1 = _mm256_extractf128_ps(max256, 1);
                                __m128 src128 = _mm_shuffle_ps(src128_0, src128_1, (2 << 2) | (2 << 6));

                                max128 = _mm_max_ps(src128, max128);
                            }
                        }
                        _mm_storeu_ps(&dst[out_index], max128);
                    }
                }

                for (; j < out_w; ++j) {
                    int out_index = j + out_w * (i + out_h * (k + c * b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i * stride + n;
                            int cur_w = w_offset + j * stride + m;
                            int index = cur_w + w * (cur_h + h * (k + b * c));
                            int valid = (cur_h >= 0 && cur_h < h&&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    if (indexes) indexes[out_index] = max_i;
                }
            }
        }
    }
}