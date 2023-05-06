#include "pch.h"
#include "conv_lstm_layer.h"
#include "utils.h"
#include "convolutional_layer.h"
#include "blas.h"
#include "activations.h"
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


layer make_conv_lstm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int peephole, int xnor, int bottleneck, int train)
{
    fprintf(stderr, "CONV_LSTM Layer: %d x %d x %d image, %d filters\n", h, w, c, output_filters);
    /*
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = outputs;
    */
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.batch = batch;
    l.type = CONV_LSTM;
    l.bottleneck = bottleneck;
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
    l.xnor = xnor;
    l.peephole = peephole;

    // U
    l.uf = (layer*)xcalloc(1, sizeof(layer));
    *(l.uf) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.uf->batch = batch;
    if (l.workspace_size < l.uf->workspace_size) l.workspace_size = l.uf->workspace_size;

    l.ui = (layer*)xcalloc(1, sizeof(layer));
    *(l.ui) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.ui->batch = batch;
    if (l.workspace_size < l.ui->workspace_size) l.workspace_size = l.ui->workspace_size;

    l.ug = (layer*)xcalloc(1, sizeof(layer));
    *(l.ug) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.ug->batch = batch;
    if (l.workspace_size < l.ug->workspace_size) l.workspace_size = l.ug->workspace_size;

    l.uo = (layer*)xcalloc(1, sizeof(layer));
    *(l.uo) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.uo->batch = batch;
    if (l.workspace_size < l.uo->workspace_size) l.workspace_size = l.uo->workspace_size;

    if (l.bottleneck) {
        // bottleneck-conv with 2x channels
        l.wf = (layer*)xcalloc(1, sizeof(layer));
        l.wi = (layer*)xcalloc(1, sizeof(layer));
        l.wg = (layer*)xcalloc(1, sizeof(layer));
        l.wo = (layer*)xcalloc(1, sizeof(layer));
        *(l.wf) = make_convolutional_layer(batch, steps, h, w, output_filters * 2, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wf->batch = batch;
        if (l.workspace_size < l.wf->workspace_size) l.workspace_size = l.wf->workspace_size;
    }
    else {
        // W
        l.wf = (layer*)xcalloc(1, sizeof(layer));
        *(l.wf) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wf->batch = batch;
        if (l.workspace_size < l.wf->workspace_size) l.workspace_size = l.wf->workspace_size;

        l.wi = (layer*)xcalloc(1, sizeof(layer));
        *(l.wi) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wi->batch = batch;
        if (l.workspace_size < l.wi->workspace_size) l.workspace_size = l.wi->workspace_size;

        l.wg = (layer*)xcalloc(1, sizeof(layer));
        *(l.wg) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wg->batch = batch;
        if (l.workspace_size < l.wg->workspace_size) l.workspace_size = l.wg->workspace_size;

        l.wo = (layer*)xcalloc(1, sizeof(layer));
        *(l.wo) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wo->batch = batch;
        if (l.workspace_size < l.wo->workspace_size) l.workspace_size = l.wo->workspace_size;
    }

    // V
    l.vf = (layer*)xcalloc(1, sizeof(layer));
    if (l.peephole) {
        *(l.vf) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.vf->batch = batch;
        if (l.workspace_size < l.vf->workspace_size) l.workspace_size = l.vf->workspace_size;
    }

    l.vi = (layer*)xcalloc(1, sizeof(layer));
    if (l.peephole) {
        *(l.vi) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.vi->batch = batch;
        if (l.workspace_size < l.vi->workspace_size) l.workspace_size = l.vi->workspace_size;
    }

    l.vo = (layer*)xcalloc(1, sizeof(layer));
    if (l.peephole) {
        *(l.vo) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.vo->batch = batch;
        if (l.workspace_size < l.vo->workspace_size) l.workspace_size = l.vo->workspace_size;
    }


    l.batch_normalize = batch_normalize;

    l.out_h = l.uo->out_h;
    l.out_w = l.uo->out_w;
    l.outputs = l.uo->outputs;
    int outputs = l.outputs;
    l.inputs = w * h * c;

    if (!l.bottleneck) assert(l.wo->outputs == l.uo->outputs);
    assert(l.wf->outputs == l.uf->outputs);

    l.output = (float*)xcalloc(outputs * batch * steps, sizeof(float));
    //l.state = (float*)xcalloc(outputs * batch, sizeof(float));

    l.forward = forward_conv_lstm_layer;
    l.update = update_conv_lstm_layer;
    l.backward = backward_conv_lstm_layer;

    l.prev_state_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.prev_cell_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.cell_cpu = (float*)xcalloc(batch * outputs * steps, sizeof(float));

    l.f_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.i_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.g_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.o_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.c_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.stored_c_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.h_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.stored_h_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.temp_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.temp2_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.temp3_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.dc_cpu = (float*)xcalloc(batch * outputs, sizeof(float));
    l.dh_cpu = (float*)xcalloc(batch * outputs, sizeof(float));

    /*
    {
        int k;
        for (k = 0; k < l.uf->n; ++k) {
            l.uf->biases[k] = 2;    // ~0.9
            l.ui->biases[k] = -22;  // ~0.1
            l.uo->biases[k] = 5;    // ~1.0
        }
#ifdef GPU
        cuda_push_array(l.uf->biases_gpu, l.uf->biases, l.n);
        cuda_push_array(l.ui->biases_gpu, l.ui->biases, l.n);
        cuda_push_array(l.uo->biases_gpu, l.uo->biases, l.n);
#endif// GPU
    }
    */

#ifdef GPU
    l.forward_gpu = forward_conv_lstm_layer_gpu;
    l.backward_gpu = backward_conv_lstm_layer_gpu;
    l.update_gpu = update_conv_lstm_layer_gpu;

    //l.state_gpu = cuda_make_array(l.state, batch*l.outputs);

    l.output_gpu = cuda_make_array(0, batch * outputs * steps);
    l.delta_gpu = cuda_make_array(0, batch * l.outputs * steps);

    l.prev_state_gpu = cuda_make_array(0, batch * outputs);
    l.prev_cell_gpu = cuda_make_array(0, batch * outputs);
    l.cell_gpu = cuda_make_array(0, batch * outputs * steps);

    l.f_gpu = cuda_make_array(0, batch * outputs);
    l.i_gpu = cuda_make_array(0, batch * outputs);
    l.g_gpu = cuda_make_array(0, batch * outputs);
    l.o_gpu = cuda_make_array(0, batch * outputs);
    l.c_gpu = cuda_make_array(0, batch * outputs);
    if (l.bottleneck) {
        l.bottelneck_hi_gpu = cuda_make_array(0, batch * outputs * 2);
        l.bottelneck_delta_gpu = cuda_make_array(0, batch * outputs * 2);
    }
    l.h_gpu = cuda_make_array(0, batch * outputs);
    l.stored_c_gpu = cuda_make_array(0, batch * outputs);
    l.stored_h_gpu = cuda_make_array(0, batch * outputs);
    l.temp_gpu = cuda_make_array(0, batch * outputs);
    l.temp2_gpu = cuda_make_array(0, batch * outputs);
    l.temp3_gpu = cuda_make_array(0, batch * outputs);
    l.dc_gpu = cuda_make_array(0, batch * outputs);
    l.dh_gpu = cuda_make_array(0, batch * outputs);
    l.last_prev_state_gpu = cuda_make_array(0, l.batch * l.outputs);
    l.last_prev_cell_gpu = cuda_make_array(0, l.batch * l.outputs);
#endif

    l.bflops = l.uf->bflops + l.ui->bflops + l.ug->bflops + l.uo->bflops +
        l.wf->bflops + l.wi->bflops + l.wg->bflops + l.wo->bflops +
        l.vf->bflops + l.vi->bflops + l.vo->bflops;

    if (l.peephole) l.bflops += 12 * l.outputs * l.batch / 1000000000.;
    else l.bflops += 9 * l.outputs * l.batch / 1000000000.;

    return l;
}


void forward_conv_lstm_layer(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    int i;
    layer vf = *(l.vf);
    layer vi = *(l.vi);
    layer vo = *(l.vo);

    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    if (state.train) {
        if (l.peephole) {
            fill_cpu(l.outputs * l.batch * l.steps, 0, vf.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, vi.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, vo.delta, 1);
        }

        fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

        fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);

        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
    }

    for (i = 0; i < l.steps; ++i)
    {
        if (l.peephole) {
            assert(l.outputs == vf.out_w * vf.out_h * vf.out_c);
            s.input = l.c_cpu;
            forward_convolutional_layer(vf, s);
            forward_convolutional_layer(vi, s);
            // vo below
        }

        assert(l.outputs == wf.out_w * wf.out_h * wf.out_c);
        assert(wf.c == l.out_c && wi.c == l.out_c && wg.c == l.out_c && wo.c == l.out_c);

        s.input = l.h_cpu;
        forward_convolutional_layer(wf, s);
        forward_convolutional_layer(wi, s);
        forward_convolutional_layer(wg, s);
        forward_convolutional_layer(wo, s);

        assert(l.inputs == uf.w * uf.h * uf.c);
        assert(uf.c == l.c && ui.c == l.c && ug.c == l.c && uo.c == l.c);

        s.input = state.input;
        forward_convolutional_layer(uf, s);
        forward_convolutional_layer(ui, s);
        forward_convolutional_layer(ug, s);
        forward_convolutional_layer(uo, s);

        // f = wf + uf + vf
        copy_cpu(l.outputs * l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uf.output, 1, l.f_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs * l.batch, 1, vf.output, 1, l.f_cpu, 1);

        // i = wi + ui + vi
        copy_cpu(l.outputs * l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ui.output, 1, l.i_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs * l.batch, 1, vi.output, 1, l.i_cpu, 1);

        // g = wg + ug
        copy_cpu(l.outputs * l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ug.output, 1, l.g_cpu, 1);

        activate_array(l.f_cpu, l.outputs * l.batch, LOGISTIC);
        activate_array(l.i_cpu, l.outputs * l.batch, LOGISTIC);
        activate_array(l.g_cpu, l.outputs * l.batch, TANH);

        // c = f*c + i*g
        copy_cpu(l.outputs * l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.c_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);

        // o = wo + uo + vo(c_new)
        if (l.peephole) {
            s.input = l.c_cpu;
            forward_convolutional_layer(vo, s);
        }
        copy_cpu(l.outputs * l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uo.output, 1, l.o_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs * l.batch, 1, vo.output, 1, l.o_cpu, 1);
        activate_array(l.o_cpu, l.outputs * l.batch, LOGISTIC);

        // h = o * tanh(c)
        copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.h_cpu, 1);
        activate_array(l.h_cpu, l.outputs * l.batch, TANH);
        mul_cpu(l.outputs * l.batch, l.o_cpu, 1, l.h_cpu, 1);

        if (l.state_constrain) constrain_cpu(l.outputs * l.batch, l.state_constrain, l.c_cpu);
        fix_nan_and_inf_cpu(l.c_cpu, l.outputs * l.batch);
        fix_nan_and_inf_cpu(l.h_cpu, l.outputs * l.batch);

        copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.cell_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.h_cpu, 1, l.output, 1);

        state.input += l.inputs * l.batch;
        l.output += l.outputs * l.batch;
        l.cell_cpu += l.outputs * l.batch;

        if (l.peephole) {
            increment_layer(&vf, 1);
            increment_layer(&vi, 1);
            increment_layer(&vo, 1);
        }

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

void backward_conv_lstm_layer(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    int i;
    layer vf = *(l.vf);
    layer vi = *(l.vi);
    layer vo = *(l.vo);

    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    if (l.peephole) {
        increment_layer(&vf, l.steps - 1);
        increment_layer(&vi, l.steps - 1);
        increment_layer(&vo, l.steps - 1);
    }

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input += l.inputs * l.batch * (l.steps - 1);
    if (state.delta) state.delta += l.inputs * l.batch * (l.steps - 1);

    l.output += l.outputs * l.batch * (l.steps - 1);
    l.cell_cpu += l.outputs * l.batch * (l.steps - 1);
    l.delta += l.outputs * l.batch * (l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_cpu(l.outputs * l.batch, l.cell_cpu - l.outputs * l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if (i != 0) copy_cpu(l.outputs * l.batch, l.output - l.outputs * l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.output, 1, l.h_cpu, 1);

        l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs * l.batch;

        // f = wf + uf + vf
        copy_cpu(l.outputs * l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uf.output, 1, l.f_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs * l.batch, 1, vf.output, 1, l.f_cpu, 1);

        // i = wi + ui + vi
        copy_cpu(l.outputs * l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ui.output, 1, l.i_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs * l.batch, 1, vi.output, 1, l.i_cpu, 1);

        // g = wg + ug
        copy_cpu(l.outputs * l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ug.output, 1, l.g_cpu, 1);

        // o = wo + uo + vo
        copy_cpu(l.outputs * l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uo.output, 1, l.o_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs * l.batch, 1, vo.output, 1, l.o_cpu, 1);

        activate_array(l.f_cpu, l.outputs * l.batch, LOGISTIC);
        activate_array(l.i_cpu, l.outputs * l.batch, LOGISTIC);
        activate_array(l.g_cpu, l.outputs * l.batch, TANH);
        activate_array(l.o_cpu, l.outputs * l.batch, LOGISTIC);

        copy_cpu(l.outputs * l.batch, l.delta, 1, l.temp3_cpu, 1);

        copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs * l.batch, TANH);

        copy_cpu(l.outputs * l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.o_cpu, 1, l.temp2_cpu, 1);

        gradient_array(l.temp_cpu, l.outputs * l.batch, TANH, l.temp2_cpu);
        axpy_cpu(l.outputs * l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);
        // temp  = tanh(c)
        // temp2 = delta * o * grad_tanh(tanh(c))
        // temp3 = delta

        copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs * l.batch, TANH);
        mul_cpu(l.outputs * l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.o_cpu, l.outputs * l.batch, LOGISTIC, l.temp_cpu);
        // delta for o(w,u,v):       temp  = delta * tanh(c) * grad_logistic(o)
        // delta for c,f,i,g(w,u,v): temp2 = delta * o * grad_tanh(tanh(c)) + delta_c(???)
        // delta for output:         temp3 = delta

        // o
        // delta for O(w,u,v):     temp  = delta * tanh(c) * grad_logistic(o)
        if (l.peephole) {
            copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, vo.delta, 1);
            s.input = l.cell_cpu;
            //s.delta = l.dc_cpu;
            backward_convolutional_layer(vo, s);
        }

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wo, s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(uo, s);

        // g
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.g_cpu, l.outputs * l.batch, TANH, l.temp_cpu);
        // delta for c,f,i,g(w,u,v): temp2 = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * g * grad_logistic(i)

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wg, s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(ug, s);

        // i
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.i_cpu, l.outputs * l.batch, LOGISTIC, l.temp_cpu);
        // delta for c,f,i,g(w,u,v): temp2 = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * g * grad_logistic(i)

        if (l.peephole) {
            copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, vi.delta, 1);
            s.input = l.prev_cell_cpu;
            //s.delta = l.dc_cpu;
            backward_convolutional_layer(vi, s);
        }

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wi, s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(ui, s);

        // f
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs * l.batch, LOGISTIC, l.temp_cpu);
        // delta for c,f,i,g(w,u,v): temp2 = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * c * grad_logistic(f)

        if (l.peephole) {
            copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, vf.delta, 1);
            s.input = l.prev_cell_cpu;
            //s.delta = l.dc_cpu;
            backward_convolutional_layer(vf, s);
        }

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wf, s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(uf, s);

        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.temp_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, l.dc_cpu, 1);

        state.input -= l.inputs * l.batch;
        if (state.delta) state.delta -= l.inputs * l.batch;
        l.output -= l.outputs * l.batch;
        l.cell_cpu -= l.outputs * l.batch;
        l.delta -= l.outputs * l.batch;

        if (l.peephole) {
            increment_layer(&vf, -1);
            increment_layer(&vi, -1);
            increment_layer(&vo, -1);
        }

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}


void update_conv_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    if (l.peephole) {
        update_convolutional_layer(*(l.vf), batch, learning_rate, momentum, decay);
        update_convolutional_layer(*(l.vi), batch, learning_rate, momentum, decay);
        update_convolutional_layer(*(l.vo), batch, learning_rate, momentum, decay);
    }
    update_convolutional_layer(*(l.wf), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.wi), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.wg), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.wo), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.uf), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.ui), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.ug), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.uo), batch, learning_rate, momentum, decay);
}


layer make_history_layer(int batch, int h, int w, int c, int history_size, int steps, int train)
{
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.batch = batch;
    l.type = HISTORY;
    l.steps = steps;
    l.history_size = history_size;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_h = h;
    l.out_w = w;
    l.out_c = c * history_size;
    l.inputs = h * w * c;
    l.outputs = h * w * c * history_size;

    l.forward = forward_history_layer;
    l.backward = backward_history_layer;

    fprintf(stderr, "HISTORY b = %d, s = %2d, steps = %2d   %4d x%4d x%4d -> %4d x%4d x%4d \n", l.batch / l.steps, l.history_size, l.steps, w, h, c, l.out_w, l.out_h, l.out_c);

    l.output = (float*)xcalloc(l.batch * l.outputs, sizeof(float));
    l.delta = (float*)xcalloc(l.batch * l.outputs, sizeof(float));

    l.prev_state_cpu = (float*)xcalloc(l.batch * l.outputs, sizeof(float));

#ifdef GPU

    l.forward_gpu = forward_history_layer_gpu;
    l.backward_gpu = backward_history_layer_gpu;

    l.output_gpu = cuda_make_array(0, l.batch * l.outputs);
    l.delta_gpu = cuda_make_array(0, l.batch * l.outputs);

    l.prev_state_gpu = cuda_make_array(0, l.batch * l.outputs);

#endif  // GPU

    //l.batch = 4;
    //l.steps = 1;

    return l;
}


void forward_history_layer(layer l, network_state state)
{
    if (l.steps == 1) {
        copy_cpu(l.inputs * l.batch, state.input, 1, l.output, 1);
        return;
    }

    const int batch = l.batch / l.steps;

    float* prev_output = l.prev_state_cpu;

    int i;
    for (i = 0; i < l.steps; ++i) {
        // shift cell
        int shift_size = l.inputs * (l.history_size - 1);
        int output_sift = l.inputs;

        int b;
        for (b = 0; b < batch; ++b) {
            int input_start = b * l.inputs + i * l.inputs * batch;
            int output_start = b * l.outputs + i * l.outputs * batch;
            float* input = state.input + input_start;
            float* output = l.output + output_start;

            copy_cpu(shift_size, prev_output + b * l.outputs, 1, output + output_sift, 1);

            copy_cpu(l.inputs, input, 1, output, 1);
        }
        prev_output = l.output + i * l.outputs * batch;
    }

    int output_start = (l.steps - 1) * l.outputs * batch;
    copy_cpu(batch * l.outputs, l.output + output_start, 1, l.prev_state_cpu, 1);
}

void backward_history_layer(layer l, network_state state)
{
    if (l.steps == 1) {
        axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, state.delta, 1);
        return;
    }

    const int batch = l.batch / l.steps;

    // l.delta -> state.delta
    int i;
    for (i = 0; i < l.steps; ++i) {
        int b;
        for (b = 0; b < batch; ++b) {
            int input_start = b * l.inputs + i * l.inputs * batch;
            int output_start = b * l.outputs + i * l.outputs * batch;
            float* state_delta = state.delta + input_start;
            float* l_delta = l.delta + output_start;

            //copy_cpu(l.inputs, l_delta, 1, state_delta, 1);
            axpy_cpu(l.inputs, 1, l_delta, 1, state_delta, 1);
        }
    }
}