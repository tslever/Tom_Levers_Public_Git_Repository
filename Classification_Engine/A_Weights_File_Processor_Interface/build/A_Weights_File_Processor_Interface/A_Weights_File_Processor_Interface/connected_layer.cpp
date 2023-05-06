#include "pch.h"
#include "connected_layer.h"
#include "utils.h"
#include <corecrt_math.h>
#include "blas.h"
#include "gemm.h"
#include "activations.h"
#include "DNLIB_Utilities.h"


connected_layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize)
{
    int total_batch = batch * steps;
    int i;
    connected_layer l = { (LAYER_TYPE)0 };
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.n = l.out_c;
    l.size = 1;
    l.stride = l.stride_x = l.stride_y = 1;
    l.pad = 0;
    l.activation = activation;
    l.learning_rate_scale = 1;
    l.groups = 1;
    l.dilation = 1;

    l.output = (float*)xcalloc(total_batch * outputs, sizeof(float));
    l.delta = (float*)xcalloc(total_batch * outputs, sizeof(float));

    l.weight_updates = (float*)xcalloc(inputs * outputs, sizeof(float));
    l.bias_updates = (float*)xcalloc(outputs, sizeof(float));

    l.weights = (float*)xcalloc(outputs * inputs, sizeof(float));
    l.biases = (float*)xcalloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2.f / inputs);
    for (i = 0; i < outputs * inputs; ++i) {
        l.weights[i] = scale * rand_uniform(-1, 1);
    }

    for (i = 0; i < outputs; ++i) {
        l.biases[i] = 0;
    }

    if (batch_normalize) {
        l.scales = (float*)xcalloc(outputs, sizeof(float));
        l.scale_updates = (float*)xcalloc(outputs, sizeof(float));
        for (i = 0; i < outputs; ++i) {
            l.scales[i] = 1;
        }

        l.mean = (float*)xcalloc(outputs, sizeof(float));
        l.mean_delta = (float*)xcalloc(outputs, sizeof(float));
        l.variance = (float*)xcalloc(outputs, sizeof(float));
        l.variance_delta = (float*)xcalloc(outputs, sizeof(float));

        l.rolling_mean = (float*)xcalloc(outputs, sizeof(float));
        l.rolling_variance = (float*)xcalloc(outputs, sizeof(float));

        l.x = (float*)xcalloc(total_batch * outputs, sizeof(float));
        l.x_norm = (float*)xcalloc(total_batch * outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs * inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs * inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs * total_batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs * total_batch);
    if (batch_normalize) {
        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.x_gpu = cuda_make_array(l.output, total_batch * outputs);
        l.x_norm_gpu = cuda_make_array(l.output, total_batch * outputs);
    }
#ifdef CUDNN
    create_convolutional_cudnn_tensors(&l);
    cudnn_convolutional_setup(&l, cudnn_fastest, 0);   // cudnn_fastest, cudnn_smallest
    l.workspace_size = get_connected_workspace_size(l);
#endif  // CUDNN
#endif  // GPU
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}


void forward_connected_layer(connected_layer l, network_state state)
{
    int i;
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float* a = state.input;
    float* b = l.weights;
    float* c = l.output;
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    if (l.batch_normalize) {
        if (state.train) {
            mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);

            scal_cpu(l.outputs, .95f, l.rolling_mean, 1);
            axpy_cpu(l.outputs, .05f, l.mean, 1, l.rolling_mean, 1);
            scal_cpu(l.outputs, .95f, l.rolling_variance, 1);
            axpy_cpu(l.outputs, .05f, l.variance, 1, l.rolling_variance, 1);

            copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);
            copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1);
        }
        else {
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);
        }
        scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
    }
    for (i = 0; i < l.batch; ++i) {
        axpy_cpu(l.outputs, 1, l.biases, 1, l.output + i * l.outputs, 1);
    }
    activate_array(l.output, l.outputs * l.batch, l.activation);
}


void backward_connected_layer(connected_layer l, network_state state)
{
    int i;
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    for (i = 0; i < l.batch; ++i) {
        axpy_cpu(l.outputs, 1, l.delta + i * l.outputs, 1, l.bias_updates, 1);
    }
    if (l.batch_normalize) {
        backward_scale_cpu(l.x_norm, l.delta, l.batch, l.outputs, 1, l.scale_updates);

        scale_bias(l.delta, l.scales, l.batch, l.outputs, 1);

        mean_delta_cpu(l.delta, l.variance, l.batch, l.outputs, 1, l.mean_delta);
        variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outputs, 1, l.variance_delta);
        normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.outputs, 1, l.delta);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float* a = l.delta;
    float* b = state.input;
    float* c = l.weight_updates;
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = state.delta;

    if (c) gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}


void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if (l.batch_normalize) {
        axpy_cpu(l.outputs, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs * l.outputs, -decay * batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs * l.outputs, momentum, l.weight_updates, 1);
}


size_t get_connected_workspace_size(layer l)
{
#ifdef CUDNN
    return get_convolutional_workspace_size(l);
    /*
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.weightDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dweightDesc,
            l.bf_algo,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.weightDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dsrcTensorDesc,
            l.bd_algo,
            &s));
        if (s > most) most = s;
        return most;
    }
    */
#endif
    return 0;
}