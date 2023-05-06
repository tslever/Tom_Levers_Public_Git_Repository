#include "pch.h"
#include "shortcut_layer.h"
#include "utils.h"
#include <corecrt_math.h>
#include "blas.h"
#include "activations.h"
#include "gemm.h"
#include "DNLIB_Utilities.h"


layer make_shortcut_layer(int batch, int n, int* input_layers, int* input_sizes, int w, int h, int c,
    float** layers_output, float** layers_delta, float** layers_output_gpu, float** layers_delta_gpu, WEIGHTS_TYPE_T weights_type, WEIGHTS_NORMALIZATION_T weights_normalization,
    ACTIVATION activation, int train)
{
    fprintf(stderr, "Shortcut Layer: ");
    int i;
    for (i = 0; i < n; ++i) fprintf(stderr, "%d, ", input_layers[i]);

    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.type = SHORTCUT;
    l.batch = batch;
    l.activation = activation;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.layers_output = layers_output;
    l.layers_delta = layers_delta;
    l.weights_type = weights_type;
    l.weights_normalization = weights_normalization;
    l.learning_rate_scale = 1;  // not necessary

    //l.w = w2;
    //l.h = h2;
    //l.c = c2;
    l.w = l.out_w = w;
    l.h = l.out_h = h;
    l.c = l.out_c = c;
    l.outputs = w * h * c;
    l.inputs = l.outputs;

    //if(w != w2 || h != h2 || c != c2) fprintf(stderr, " w = %d, w2 = %d, h = %d, h2 = %d, c = %d, c2 = %d \n", w, w2, h, h2, c, c2);

    l.index = l.input_layers[0];


    if (train) l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

    l.nweights = 0;
    if (l.weights_type == PER_FEATURE) l.nweights = (l.n + 1);
    else if (l.weights_type == PER_CHANNEL) l.nweights = (l.n + 1) * l.c;

    if (l.nweights > 0) {
        l.weights = (float*)calloc(l.nweights, sizeof(float));
        float scale = sqrt(2. / l.nweights);
        for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;// +0.01*rand_uniform(-1, 1);// scale*rand_uniform(-1, 1);   // rand_normal();

        if (train) l.weight_updates = (float*)calloc(l.nweights, sizeof(float));
        l.update = update_shortcut_layer;
    }

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
#ifndef GPU
    if (l.activation == SWISH || l.activation == MISH) l.activation_input = (float*)calloc(l.batch * l.outputs, sizeof(float));
#endif // GPU

#ifdef GPU
    if (l.activation == SWISH || l.activation == MISH) l.activation_input_gpu = cuda_make_array(l.activation_input, l.batch * l.outputs);

    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    if (l.nweights > 0) {
        l.update_gpu = update_shortcut_layer_gpu;
        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        if (train) l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
    }

    if (train) l.delta_gpu = cuda_make_array(l.delta, l.outputs * batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs * batch);

    l.input_sizes_gpu = cuda_make_int_array_new_api(input_sizes, l.n);
    l.layers_output_gpu = (float**)cuda_make_array_pointers((void**)layers_output_gpu, l.n);
    l.layers_delta_gpu = (float**)cuda_make_array_pointers((void**)layers_delta_gpu, l.n);
#endif  // GPU

    l.bflops = l.out_w * l.out_h * l.out_c * l.n / 1000000000.;
    if (l.weights_type) l.bflops *= 2;
    fprintf(stderr, " wt = %d, wn = %d, outputs:%4d x%4d x%4d %5.3f BF\n", l.weights_type, l.weights_normalization, l.out_w, l.out_h, l.out_c, l.bflops);
    return l;
}


void forward_shortcut_layer(const layer l, network_state state)
{
    int from_w = state.net.layers[l.index].w;
    int from_h = state.net.layers[l.index].h;
    int from_c = state.net.layers[l.index].c;

    if (l.nweights == 0 && l.n == 1 && from_w == l.w && from_h == l.h && from_c == l.c) {
        int size = l.batch * l.w * l.h * l.c;
        int i;
#pragma omp parallel for
        for (i = 0; i < size; ++i)
            l.output[i] = state.input[i] + state.net.layers[l.index].output[i];
    }
    else {
        shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes, l.layers_output, l.output, state.input, l.weights, l.nweights, l.weights_normalization);
    }

    //copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    //shortcut_cpu(l.batch, from_w, from_h, from_c, state.net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);

    //activate_array(l.output, l.outputs*l.batch, l.activation);
    if (l.activation == SWISH) activate_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.output);
    else if (l.activation == MISH) activate_array_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
    else activate_array_cpu_custom(l.output, l.outputs * l.batch, l.activation);
}


void backward_shortcut_layer(const layer l, network_state state)
{
    if (l.activation == SWISH) gradient_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.delta);
    else if (l.activation == MISH) gradient_array_mish(l.outputs * l.batch, l.activation_input, l.delta);
    else gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

    backward_shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes,
        l.layers_delta, state.delta, l.delta, l.weights, l.weight_updates, l.nweights, state.input, l.layers_output, l.weights_normalization);

    //axpy_cpu(l.outputs*l.batch, 1, l.delta, 1, state.delta, 1);
    //shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, state.net.layers[l.index].delta);
}


void update_shortcut_layer(layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    if (l.nweights > 0) {
        float learning_rate = learning_rate_init * l.learning_rate_scale;
        //float momentum = a.momentum;
        //float decay = a.decay;
        //int batch = a.batch;

        axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
        axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
        scal_cpu(l.nweights, momentum, l.weight_updates, 1);
    }
}