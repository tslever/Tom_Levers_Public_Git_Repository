#ifndef BLAS_H
#define BLAS_H


#include "darknet.h"


void fill_cpu(int N, float ALPHA, float* X, int INCX);
void copy_cpu(int N, float* X, int INCX, float* Y, int INCY);
void mean_cpu(float* x, int batch, int filters, int spatial, float* mean);
void variance_cpu(float* x, float* mean, int batch, int filters, int spatial, float* variance);
void scal_cpu(int N, float ALPHA, float* X, int INCX);
void axpy_cpu(int N, float ALPHA, float* X, int INCX, float* Y, int INCY);
void normalize_cpu(float* x, float* mean, float* variance, int batch, int filters, int spatial);
void scale_bias(float* output, float* scales, int batch, int n, int size);
void add_bias(float* output, float* biases, int batch, int n, int size);
void backward_scale_cpu(float* x_norm, float* delta, int batch, int n, int size, float* scale_updates);
void mean_delta_cpu(float* delta, float* variance, int batch, int filters, int spatial, float* mean_delta);
void  variance_delta_cpu(float* x, float* delta, float* mean, float* variance, int batch, int filters, int spatial, float* variance_delta);
void normalize_delta_cpu(float* x, float* mean, float* variance, float* mean_delta, float* variance_delta, int batch, int filters, int spatial, float* delta);
void mul_cpu(int N, float* X, int INCX, float* Y, int INCY);
void weighted_sum_cpu(float* a, float* b, float* s, int num, float* c);
void constrain_cpu(int size, float ALPHA, float* X);
void fix_nan_and_inf_cpu(float* input, size_t size);
void smooth_l1_cpu(int n, float* pred, float* truth, float* delta, float* error);
void l2_cpu(int n, float* pred, float* truth, float* delta, float* error);
void flatten(float* x, int size, int layers, int batch, int forward);
void softmax(float* input, int n, float temp, float* output, int stride);
void scal_add_cpu(int N, float ALPHA, float BETA, float* X, int INCX);
void softmax_cpu(float* input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float* output);
void softmax_x_ent_cpu(int n, float* pred, float* truth, float* delta, float* error);
void get_embedding(float* src, int src_w, int src_h, int src_c, int embedding_size, int cur_w, int cur_h, int cur_n, int cur_b, float* dst);
float cosine_similarity(float* A, float* B, unsigned int feature_size);
float P_constrastive_f_det(size_t il, int* labels, float** z, unsigned int feature_size, float temperature, contrastive_params* contrast_p, int contrast_p_size);
float P_constrastive_f(size_t i, size_t l, int* labels, float** z, unsigned int feature_size, float temperature, contrastive_params* contrast_p, int contrast_p_size);
float P_constrastive(size_t i, size_t l, int* labels, size_t num_of_samples, float** z, unsigned int feature_size, float temperature, float* cos_sim, float* exp_cos_sim);
void scal_cpu(int N, float ALPHA, float* X, int INCX);
void pow_cpu(int N, float ALPHA, float* X, int INCX, float* Y, int INCY);
void const_cpu(int N, float ALPHA, float* X, int INCX);
void reorg_cpu(float* x, int w, int h, int c, int batch, int stride, int forward, float* out);
void upsample_cpu(float* in, int w, int h, int c, int batch, int stride, int forward, float scale, float* out);
void shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int* outputs_of_layers, float** layers_output, float* out, float* in, float* weights, int nweights, WEIGHTS_NORMALIZATION_T weights_normalization);
void backward_shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int* outputs_of_layers,
    float** layers_delta, float* delta_out, float* delta_in, float* weights, float* weight_updates, int nweights, float* in, float** layers_output, WEIGHTS_NORMALIZATION_T weights_normalization);
float find_sim(size_t i, size_t j, contrastive_params* contrast_p, int contrast_p_size);
void grad_contrastive_loss_positive_f(size_t i, int* class_ids, int* labels, size_t num_of_samples, float** z, unsigned int feature_size, float temperature, float* delta, int wh, contrastive_params* contrast_p, int contrast_p_size);
float math_vector_length(float* A, unsigned int feature_size);
void grad_contrastive_loss_negative_f(size_t i, int* class_ids, int* labels, size_t num_of_samples, float** z, unsigned int feature_size, float temperature, float* delta, int wh, contrastive_params* contrast_p, int contrast_p_size, int neg_max);
void grad_contrastive_loss_positive(size_t i, int* labels, size_t num_of_samples, float** z, unsigned int feature_size, float temperature, float* cos_sim, float* p_constrastive, float* delta, int wh);
void grad_contrastive_loss_negative(size_t i, int* labels, size_t num_of_samples, float** z, unsigned int feature_size, float temperature, float* cos_sim, float* p_constrastive, float* delta, int wh);


#endif