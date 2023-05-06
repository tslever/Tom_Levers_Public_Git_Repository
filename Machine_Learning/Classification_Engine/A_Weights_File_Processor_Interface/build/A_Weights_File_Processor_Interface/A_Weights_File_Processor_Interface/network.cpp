#include "pch.h"
#include "network.h"
#include "utils.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include <float.h>
#include <corecrt_math.h>
#include "DNLIB_Utilities.h"


int get_current_batch(network net)
{
    int batch_num = (*net.seen) / (net.batch * net.subdivisions);
    return batch_num;
}


network make_network(int n)
{
    network net = { 0 };
    net.n = n;
    net.layers = (layer*)xcalloc(net.n, sizeof(layer));
    net.seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));
    net.cuda_graph_ready = (int*)xcalloc(1, sizeof(int));
    net.badlabels_reject_threshold = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_max = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_avg = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_std = (float*)xcalloc(1, sizeof(float));
    net.cur_iteration = (int*)xcalloc(1, sizeof(int));
    net.total_bbox = (int*)xcalloc(1, sizeof(int));
    net.rewritten_bbox = (int*)xcalloc(1, sizeof(int));
    *net.rewritten_bbox = *net.total_bbox = 0;
#ifdef GPU
    net.input_gpu = (float**)xcalloc(1, sizeof(float*));
    net.truth_gpu = (float**)xcalloc(1, sizeof(float*));

    net.input16_gpu = (float**)xcalloc(1, sizeof(float*));
    net.output16_gpu = (float**)xcalloc(1, sizeof(float*));
    net.max_input16_size = (size_t*)xcalloc(1, sizeof(size_t));
    net.max_output16_size = (size_t*)xcalloc(1, sizeof(size_t));
#endif
    return net;
}


int get_network_output_size(network net)
{
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}


float* get_network_output(network net)
{
#ifdef GPU

    // MODIFIED BY TSL ON 08/25/21 00:54
    int gpu_index = 0;

    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}


int64_t get_current_iteration(network net)
{
    return *net.cur_iteration;
}


int recalculate_workspace_size(network* net)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);

    // MODIFIED BY TSL ON 08/25/21 00:54
    int gpu_index = 0;

    if (gpu_index >= 0) cuda_free(net->workspace);
#endif
    int i;
    size_t workspace_size = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        //printf(" %d: layer = %d,", i, l.type);
        if (l.type == CONVOLUTIONAL) {
            l.workspace_size = get_convolutional_workspace_size(l);
        }
        else if (l.type == CONNECTED) {
            l.workspace_size = get_connected_workspace_size(l);
        }
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        net->layers[i] = l;
    }

#ifdef GPU

    // MODIFIED BY TSL ON 08/25/21 00:54
    int gpu_index = 0;

    if (gpu_index >= 0) {
        printf("\n try to allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
        net->workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        printf(" CUDA allocate done! \n");
    }
    else {
        free(net->workspace);
        net->workspace = (float*)xcalloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}


void set_batch_network(network* net, int b)
{
    net->batch = b;
    int i;
    for (i = 0; i < net->n; ++i) {
        net->layers[i].batch = b;

#ifdef CUDNN
        if (net->layers[i].type == CONVOLUTIONAL) {
            cudnn_convolutional_setup(net->layers + i, cudnn_fastest, 0);
        }
        else if (net->layers[i].type == MAXPOOL) {
            cudnn_maxpool_setup(net->layers + i);
        }
#endif

    }
    recalculate_workspace_size(net); // recalculate workspace size
}


static float lrelu(float src) {
    const float eps = 0.001;
    if (src > eps) return src;
    return eps;
}


void fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer* l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->share_layer != NULL) {
                l->batch_normalize = 0;
            }

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    double precomputed = l->scales[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    const size_t filter_size = l->size * l->size * l->c / l->groups;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f * filter_size + i;

                        l->weights[w_index] *= precomputed;
                    }
                }

                free_convolutional_batchnorm(l);
                l->batch_normalize = 0;
#ifdef GPU
                // MODIFIED BY TSL ON 08/25/21 00:54
                int gpu_index = 0;

                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else  if (l->type == SHORTCUT && l->weights && l->weights_normalization)
        {
            if (l->nweights > 0) {
                //cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
                int i;
                for (i = 0; i < l->nweights; ++i) printf(" w = %f,", l->weights[i]);
                printf(" l->nweights = %d, j = %d \n", l->nweights, j);
            }

            // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
            const int layer_step = l->nweights / (l->n + 1);    // 1 or l.c or (l.c * l.h * l.w)

            int chan, i;
            for (chan = 0; chan < layer_step; ++chan)
            {
                float sum = 1, max_val = -FLT_MAX;

                if (l->weights_normalization == SOFTMAX_NORMALIZATION) {
                    for (i = 0; i < (l->n + 1); ++i) {
                        int w_index = chan + i * layer_step;
                        float w = l->weights[w_index];
                        if (max_val < w) max_val = w;
                    }
                }

                const float eps = 0.0001;
                sum = eps;

                for (i = 0; i < (l->n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l->weights[w_index];
                    if (l->weights_normalization == RELU_NORMALIZATION) sum += lrelu(w);
                    else if (l->weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
                }

                for (i = 0; i < (l->n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l->weights[w_index];
                    if (l->weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
                    else if (l->weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;
                    l->weights[w_index] = w;
                }
            }

            l->weights_normalization = NO_NORMALIZATION;

#ifdef GPU

            // MODIFIED BY TSL ON 08/25/21 00:54
            int gpu_index = 0;

            if (gpu_index >= 0) {
                push_shortcut_layer(*l);
            }
#endif
        }
        else {
            //printf(" Fusion skip layer type: %d \n", l->type);
        }
    }
}


void calculate_binary_weights(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer* l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->xnor) {
                //printf("\n %d \n", j);
                //l->lda_align = 256; // 256bit for AVX2    // set in make_convolutional_layer()
                //if (l->size*l->size*l->c >= 2048) l->lda_align = 512;

                binary_align_weights(l);

                if (net.layers[j].use_bin_output) {
                    l->activation = LINEAR;
                }

#ifdef GPU
                // fuse conv_xnor + shortcut -> conv_xnor
                if ((j + 1) < net.n && net.layers[j].type == CONVOLUTIONAL) {
                    layer* sc = &net.layers[j + 1];
                    if (sc->type == SHORTCUT && sc->w == sc->out_w && sc->h == sc->out_h && sc->c == sc->out_c)
                    {
                        l->bin_conv_shortcut_in_gpu = net.layers[net.layers[j + 1].index].output_gpu;
                        l->bin_conv_shortcut_out_gpu = net.layers[j + 1].output_gpu;

                        net.layers[j + 1].type = BLANK;
                        net.layers[j + 1].forward_gpu = forward_blank_layer;
                    }
                }
#endif  // GPU
            }
        }
    }
    //printf("\n calculate_binary_weights Done! \n");

}