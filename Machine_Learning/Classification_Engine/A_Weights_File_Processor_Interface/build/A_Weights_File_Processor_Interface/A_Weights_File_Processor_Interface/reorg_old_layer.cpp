#include "pch.h"
#include "reorg_old_layer.h"
#include "utils.h"
#include "blas.h"
#include "DNLIB_Utilities.h"


layer make_reorg_old_layer(int batch, int w, int h, int c, int stride, int reverse)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = REORG_OLD;
    l.batch = batch;
    l.stride = stride;
    l.h = h;
    l.w = w;
    l.c = c;
    if (reverse) {
        l.out_w = w * stride;
        l.out_h = h * stride;
        l.out_c = c / (stride * stride);
    }
    else {
        l.out_w = w / stride;
        l.out_h = h / stride;
        l.out_c = c * (stride * stride);
    }
    l.reverse = reverse;
    fprintf(stderr, "reorg_old              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h * w * c;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.output = (float*)xcalloc(output_size, sizeof(float));
    l.delta = (float*)xcalloc(output_size, sizeof(float));

    l.forward = forward_reorg_old_layer;
    l.backward = backward_reorg_old_layer;
#ifdef GPU
    l.forward_gpu = forward_reorg_old_layer_gpu;
    l.backward_gpu = backward_reorg_old_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, output_size);
    l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif
    return l;
}


void forward_reorg_old_layer(const layer l, network_state state)
{
    if (l.reverse) {
        reorg_cpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
    }
    else {
        reorg_cpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
    }
}


void backward_reorg_old_layer(const layer l, network_state state)
{
    if (l.reverse) {
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, state.delta);
    }
    else {
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, state.delta);
    }
}