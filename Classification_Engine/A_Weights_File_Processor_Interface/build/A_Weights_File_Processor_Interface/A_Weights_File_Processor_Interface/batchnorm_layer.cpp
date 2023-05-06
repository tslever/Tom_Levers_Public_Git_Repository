#include "pch.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "utils.h"
#include "DNLIB_Utilities.h"


void forward_batchnorm_layer(layer l, network_state state)
{
    if (l.type == BATCHNORM) copy_cpu(l.outputs * l.batch, state.input, 1, l.output, 1);
    if (l.type == CONNECTED) {
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
    if (state.train) {
        mean_cpu(l.output, l.batch, l.out_c, l.out_h * l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h * l.out_w, l.variance);

        scal_cpu(l.out_c, .9, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .1, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .9, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .1, l.variance, 1, l.rolling_variance, 1);

        copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h * l.out_w);
        copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1);
    }
    else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h * l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h * l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_w * l.out_h);
}


void backward_batchnorm_layer(const layer l, network_state state)
{
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w * l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h * l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w * l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w * l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w * l.out_h, l.delta);
    if (l.type == BATCHNORM) copy_cpu(l.outputs * l.batch, l.delta, 1, state.delta, 1);
}


layer make_batchnorm_layer(int batch, int w, int h, int c, int train)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w, h, c);
    layer layer = { (LAYER_TYPE)0 };
    layer.type = BATCHNORM;
    layer.batch = batch;
    layer.train = train;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;

    layer.n = layer.c;
    layer.output = (float*)xcalloc(h * w * c * batch, sizeof(float));
    layer.delta = (float*)xcalloc(h * w * c * batch, sizeof(float));
    layer.inputs = w * h * c;
    layer.outputs = layer.inputs;

    layer.biases = (float*)xcalloc(c, sizeof(float));
    layer.bias_updates = (float*)xcalloc(c, sizeof(float));

    layer.scales = (float*)xcalloc(c, sizeof(float));
    layer.scale_updates = (float*)xcalloc(c, sizeof(float));
    int i;
    for (i = 0; i < c; ++i) {
        layer.scales[i] = 1;
    }

    layer.mean = (float*)xcalloc(c, sizeof(float));
    layer.variance = (float*)xcalloc(c, sizeof(float));

    layer.rolling_mean = (float*)xcalloc(c, sizeof(float));
    layer.rolling_variance = (float*)xcalloc(c, sizeof(float));

    layer.mean_delta = (float*)xcalloc(c, sizeof(float));
    layer.variance_delta = (float*)xcalloc(c, sizeof(float));

    layer.x = (float*)xcalloc(layer.batch * layer.outputs, sizeof(float));
    layer.x_norm = (float*)xcalloc(layer.batch * layer.outputs, sizeof(float));

    layer.forward = forward_batchnorm_layer;
    layer.backward = backward_batchnorm_layer;
    layer.update = update_batchnorm_layer;
#ifdef GPU
    layer.forward_gpu = forward_batchnorm_layer_gpu;
    layer.backward_gpu = backward_batchnorm_layer_gpu;
    layer.update_gpu = update_batchnorm_layer_gpu;

    layer.output_gpu = cuda_make_array(layer.output, h * w * c * batch);

    layer.biases_gpu = cuda_make_array(layer.biases, c);
    layer.scales_gpu = cuda_make_array(layer.scales, c);

    if (train) {
        layer.delta_gpu = cuda_make_array(layer.delta, h * w * c * batch);

        layer.bias_updates_gpu = cuda_make_array(layer.bias_updates, c);
        layer.scale_updates_gpu = cuda_make_array(layer.scale_updates, c);

        layer.mean_delta_gpu = cuda_make_array(layer.mean, c);
        layer.variance_delta_gpu = cuda_make_array(layer.variance, c);
    }

    layer.mean_gpu = cuda_make_array(layer.mean, c);
    layer.variance_gpu = cuda_make_array(layer.variance, c);

    layer.rolling_mean_gpu = cuda_make_array(layer.mean, c);
    layer.rolling_variance_gpu = cuda_make_array(layer.variance, c);

    if (train) {
        layer.x_gpu = cuda_make_array(layer.output, layer.batch * layer.outputs);
#ifndef CUDNN
        layer.x_norm_gpu = cuda_make_array(layer.output, layer.batch * layer.outputs);
#endif  // not CUDNN
    }

#ifdef CUDNN
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.normTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.normDstTensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(layer.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, layer.batch, layer.out_c, layer.out_h, layer.out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(layer.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, layer.out_c, 1, 1));
#endif
#endif
    return layer;
}


void update_batchnorm_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    //int size = l.nweights;
    axpy_cpu(l.c, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.c, momentum, l.bias_updates, 1);

    axpy_cpu(l.c, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
    scal_cpu(l.c, momentum, l.scale_updates, 1);
}