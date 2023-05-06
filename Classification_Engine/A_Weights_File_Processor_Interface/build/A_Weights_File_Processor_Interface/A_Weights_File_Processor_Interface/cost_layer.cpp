#include "pch.h"
#include "cost_layer.h"
#include "utils.h"
#include "blas.h"
#include "DNLIB_Utilities.h"


COST_TYPE get_cost_type(char* s)
{
    if (strcmp(s, "sse") == 0) return SSE;
    if (strcmp(s, "masked") == 0) return MASKED;
    if (strcmp(s, "smooth") == 0) return SMOOTH;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}


cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "cost                                           %4d\n", inputs);
    cost_layer l = { (LAYER_TYPE)0 };
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;
#ifdef GPU
    l.forward_gpu = forward_cost_layer_gpu;
    l.backward_gpu = backward_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.delta, inputs * batch);
    l.output_gpu = cuda_make_array(l.output, inputs * batch);
#endif
    return l;
}


void forward_cost_layer(cost_layer l, network_state state)
{
    if (!state.truth) return;
    if (l.cost_type == MASKED) {
        int i;
        for (i = 0; i < l.batch * l.inputs; ++i) {
            if (state.truth[i] == SECRET_NUM) state.input[i] = SECRET_NUM;
        }
    }
    if (l.cost_type == SMOOTH) {
        smooth_l1_cpu(l.batch * l.inputs, state.input, state.truth, l.delta, l.output);
    }
    else {
        l2_cpu(l.batch * l.inputs, state.input, state.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array(l.output, l.batch * l.inputs);
}


void backward_cost_layer(const cost_layer l, network_state state)
{
    axpy_cpu(l.batch * l.inputs, l.scale, l.delta, 1, state.delta, 1);
}