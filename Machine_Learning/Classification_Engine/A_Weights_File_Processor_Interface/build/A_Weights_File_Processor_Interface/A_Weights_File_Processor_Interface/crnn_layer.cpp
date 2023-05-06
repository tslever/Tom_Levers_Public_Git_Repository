#include "pch.h"
#include "crnn_layer.h"
#include "utils.h"
#include "convolutional_layer.h"
#include "blas.h"
#include "DNLIB_Utilities.h"


static void increment_layer(layer* l, int steps)
{
    int num = l->outputs * l->batch * steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}


layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int xnor, int train)
{
    fprintf(stderr, "CRNN Layer: %d x %d x %d image, %d filters\n", h, w, c, output_filters);
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.batch = batch;
    l.type = CRNN;
    l.steps = steps;
    l.size = size;
    l.stride = stride;
    l.dilation = dilation;
    l.pad = pad;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.out_c = output_filters;
    l.inputs = h * w * c;
    l.hidden = h * w * hidden_filters;
    l.xnor = xnor;

    l.state = (float*)xcalloc(l.hidden * l.batch * (l.steps + 1), sizeof(float));

    l.input_layer = (layer*)xcalloc(1, sizeof(layer));
    *(l.input_layer) = make_convolutional_layer(batch, steps, h, w, c, hidden_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.input_layer->batch = batch;
    if (l.workspace_size < l.input_layer->workspace_size) l.workspace_size = l.input_layer->workspace_size;

    l.self_layer = (layer*)xcalloc(1, sizeof(layer));
    *(l.self_layer) = make_convolutional_layer(batch, steps, h, w, hidden_filters, hidden_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.self_layer->batch = batch;
    if (l.workspace_size < l.self_layer->workspace_size) l.workspace_size = l.self_layer->workspace_size;

    l.output_layer = (layer*)xcalloc(1, sizeof(layer));
    *(l.output_layer) = make_convolutional_layer(batch, steps, h, w, hidden_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.output_layer->batch = batch;
    if (l.workspace_size < l.output_layer->workspace_size) l.workspace_size = l.output_layer->workspace_size;

    l.out_h = l.output_layer->out_h;
    l.out_w = l.output_layer->out_w;
    l.outputs = l.output_layer->outputs;

    assert(l.input_layer->outputs == l.self_layer->outputs);
    assert(l.input_layer->outputs == l.output_layer->inputs);

    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;

    l.forward = forward_crnn_layer;
    l.backward = backward_crnn_layer;
    l.update = update_crnn_layer;

#ifdef GPU
    l.forward_gpu = forward_crnn_layer_gpu;
    l.backward_gpu = backward_crnn_layer_gpu;
    l.update_gpu = update_crnn_layer_gpu;
    l.state_gpu = cuda_make_array(l.state, l.batch * l.hidden * (l.steps + 1));
    l.output_gpu = l.output_layer->output_gpu;
    l.delta_gpu = l.output_layer->delta_gpu;
#endif

    l.bflops = l.input_layer->bflops + l.self_layer->bflops + l.output_layer->bflops;

    return l;
}


void forward_crnn_layer(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    //s.index = state.index;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    if (state.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
        fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
        fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
        fill_cpu(l.hidden * l.batch, 0, l.state, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = state.input;
        forward_convolutional_layer(input_layer, s);

        s.input = l.state;
        forward_convolutional_layer(self_layer, s);

        float* old_state = l.state;
        if (state.train) l.state += l.hidden * l.batch;
        if (l.shortcut) {
            copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1);
        }
        else {
            fill_cpu(l.hidden * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_convolutional_layer(output_layer, s);

        state.input += l.inputs * l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_crnn_layer(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    //s.index = state.index;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    increment_layer(&input_layer, l.steps - 1);
    increment_layer(&self_layer, l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);

    l.state += l.hidden * l.batch * l.steps;
    for (i = l.steps - 1; i >= 0; --i) {
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_convolutional_layer(output_layer, s);

        l.state -= l.hidden * l.batch;
        /*
           if(i > 0){
           copy_cpu(l.hidden * l.batch, input_layer.output - l.hidden*l.batch, 1, l.state, 1);
           axpy_cpu(l.hidden * l.batch, 1, self_layer.output - l.hidden*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.hidden * l.batch, 0, l.state, 1);
           }
         */

        s.input = l.state;
        s.delta = self_layer.delta - l.hidden * l.batch;
        if (i == 0) s.delta = 0;
        backward_convolutional_layer(self_layer, s);

        copy_cpu(l.hidden * l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.hidden * l.batch, 1, self_layer.delta, 1, self_layer.delta - l.hidden * l.batch, 1);
        s.input = state.input + i * l.inputs * l.batch;
        if (state.delta) s.delta = state.delta + i * l.inputs * l.batch;
        else s.delta = 0;
        backward_convolutional_layer(input_layer, s);

        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}


void update_crnn_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    update_convolutional_layer(*(l.input_layer), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.self_layer), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.output_layer), batch, learning_rate, momentum, decay);
}