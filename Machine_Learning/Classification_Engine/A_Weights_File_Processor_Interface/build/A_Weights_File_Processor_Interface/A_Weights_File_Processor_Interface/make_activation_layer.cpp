#include "pch.h"
#include "activation_layer.h"
#include "utils.h"
#include "blas.h"
#include "activations.h"
#include "DNLIB_Utilities.h"


layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;

    l.output = (float*)xcalloc(batch * inputs, sizeof(float));
    l.delta = (float*)xcalloc(batch * inputs, sizeof(float));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs * batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs * batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}


void forward_activation_layer(layer l, network_state state)
{
    copy_cpu(l.outputs * l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs * l.batch, l.activation);
}


void backward_activation_layer(layer l, network_state state)
{
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    copy_cpu(l.outputs * l.batch, l.delta, 1, state.delta, 1);
}