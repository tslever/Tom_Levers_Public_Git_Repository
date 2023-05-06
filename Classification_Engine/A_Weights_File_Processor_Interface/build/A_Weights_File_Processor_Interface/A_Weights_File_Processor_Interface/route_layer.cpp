#include "pch.h"
#include "route_layer.h"
#include "utils.h"
#include "blas.h"
#include "DNLIB_Utilities.h"


route_layer make_route_layer(int batch, int n, int* input_layers, int* input_sizes, int groups, int group_id)
{
    fprintf(stderr, "route ");
    route_layer l = { (LAYER_TYPE)0 };
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.groups = groups;
    l.group_id = group_id;
    int i;
    int outputs = 0;
    for (i = 0; i < n; ++i) {
        fprintf(stderr, " %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    outputs = outputs / groups;
    l.outputs = outputs;
    l.inputs = outputs;
    //fprintf(stderr, " inputs = %d \t outputs = %d, groups = %d, group_id = %d \n", l.inputs, l.outputs, l.groups, l.group_id);
    l.delta = (float*)xcalloc(outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(outputs * batch, sizeof(float));

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
#ifdef GPU
    l.forward_gpu = forward_route_layer_gpu;
    l.backward_gpu = backward_route_layer_gpu;

    l.delta_gpu = cuda_make_array(l.delta, outputs * batch);
    l.output_gpu = cuda_make_array(l.output, outputs * batch);
#endif
    return l;
}


void forward_route_layer(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for (i = 0; i < l.n; ++i) {
        int index = l.input_layers[i];
        float* input = state.net.layers[index].output;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for (j = 0; j < l.batch; ++j) {
            //copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
            copy_cpu(part_input_size, input + j * input_size + part_input_size * l.group_id, 1, l.output + offset + j * l.outputs, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}


void backward_route_layer(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for (i = 0; i < l.n; ++i) {
        int index = l.input_layers[i];
        float* delta = state.net.layers[index].delta;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for (j = 0; j < l.batch; ++j) {
            //axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
            axpy_cpu(part_input_size, 1, l.delta + offset + j * l.outputs, 1, delta + j * input_size + part_input_size * l.group_id, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}