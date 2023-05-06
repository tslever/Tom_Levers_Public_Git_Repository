#include "pch.h"
#include "region_layer.h"
#include "utils.h"
#include "darknet.h"
#include "blas.h"
#include "activations.h"
#include "softmax_layer.h"
#include "box.h"
#include "tree.h"
#include "DNLIB_Utilities.h"


#define DOABS 1


region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes)
{
    region_layer l = { (LAYER_TYPE)0 };
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.biases = (float*)xcalloc(n * 2, sizeof(float));
    l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
    l.outputs = h * w * n * (classes + coords + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truth_size = 4 + 2;
    l.truths = max_boxes * l.truth_size;
    l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    int i;
    for (i = 0; i < n * 2; ++i) {
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(time(0));

    return l;
}


void delta_region_class(float* output, float* delta, int index, int class_id, int classes, tree* hier, float scale, float* avg_cat, int focal_loss)
{
    int i, n;
    if (hier) {
        float pred = 1;
        while (class_id >= 0) {
            pred *= output[index + class_id];
            int g = hier->group[class_id];
            int offset = hier->group_offset[g];
            for (i = 0; i < hier->group_size[g]; ++i) {
                delta[index + offset + i] = scale * (0 - output[index + offset + i]);
            }
            delta[index + class_id] = scale * (1 - output[index + class_id]);

            class_id = hier->parent[class_id];
        }
        *avg_cat += pred;
    }
    else {
        // Focal loss
        if (focal_loss) {
            // Focal Loss
            float alpha = 0.5;    // 0.25 or 0.5
            //float gamma = 2;    // hardcoded in many places of the grad-formula

            int ti = index + class_id;
            float pt = output[ti] + 0.000000000000001F;
            // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
            float grad = -(1 - pt) * (2 * pt * logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
            //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);

                delta[index + n] *= alpha * grad;

                if (n == class_id) *avg_cat += output[index + n];
            }
        }
        else {
            // default
            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);
                if (n == class_id) *avg_cat += output[index + n];
            }
        }
    }
}


box get_region_box(float* x, float* biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2 * n];
    b.h = exp(x[index + 3]) * biases[2 * n + 1];
    if (DOABS) {
        b.w = exp(x[index + 2]) * biases[2 * n] / w;
        b.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
    }
    return b;
}


float delta_region_box(box truth, float* x, float* biases, int n, int index, int i, int j, int w, int h, float* delta, float scale)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    float tx = (truth.x * w - i);
    float ty = (truth.y * h - j);
    float tw = log(truth.w / biases[2 * n]);
    float th = log(truth.h / biases[2 * n + 1]);
    if (DOABS) {
        tw = log(truth.w * w / biases[2 * n]);
        th = log(truth.h * h / biases[2 * n + 1]);
    }

    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
    return iou;
}


void forward_region_layer(const region_layer l, network_state state)
{
    int i, j, b, t, n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs * l.batch * sizeof(float));
#ifndef GPU
    flatten(l.output, l.w * l.h, size * l.n, l.batch, 1);
#endif
    for (b = 0; b < l.batch; ++b) {
        for (i = 0; i < l.h * l.w * l.n; ++i) {
            int index = size * i + b * l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);
        }
    }


#ifndef GPU
    if (l.softmax_tree) {
        for (b = 0; b < l.batch; ++b) {
            for (i = 0; i < l.h * l.w * l.n; ++i) {
                int index = size * i + b * l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    }
    else if (l.softmax) {
        for (b = 0; b < l.batch; ++b) {
            for (i = 0; i < l.h * l.w * l.n; ++i) {
                int index = size * i + b * l.outputs;
                softmax(l.output + index + 5, l.classes, 1, l.output + index + 5, 1);
            }
        }
    }
#endif
    if (!state.train) return;
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if (l.softmax_tree) {
            int onlyclass_id = 0;
            for (t = 0; t < l.max_boxes; ++t) {
                box truth = float_to_box(state.truth + t * l.truth_size + b * l.truths);
                if (!truth.x) break; // continue;
                int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if (truth.x > 100000 && truth.y > 100000) {
                    for (n = 0; n < l.n * l.w * l.h; ++n) {
                        int index = size * n + b * l.outputs + 5;
                        float scale = l.output[index - 1];
                        float p = scale * get_hierarchy_probability(l.output + index, l.softmax_tree, class_id);
                        if (p > maxp) {
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int index = size * maxi + b * l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
                    ++class_count;
                    onlyclass_id = 1;
                    break;
                }
            }
            if (onlyclass_id) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = size * (j * l.w * l.n + i * l.n + n) + b * l.outputs;
                    box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                    float best_iou = 0;
                    int best_class_id = -1;
                    for (t = 0; t < l.max_boxes; ++t) {
                        box truth = float_to_box(state.truth + t * l.truth_size + b * l.truths);
                        int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                        if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file
                        if (!truth.x) break; // continue;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += l.output[index + 4];
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    if (l.classfix == -1) l.delta[index + 4] = l.noobject_scale * ((best_iou - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    else {
                        if (best_iou > l.thresh) {
                            l.delta[index + 4] = 0;
                            if (l.classfix > 0) {
                                delta_region_class(l.output, l.delta, index + 5, best_class_id, l.classes, l.softmax_tree, l.class_scale * (l.classfix == 2 ? l.output[index + 4] : 1), &avg_cat, l.focal_loss);
                                ++class_count;
                            }
                        }
                    }

                    if (*(state.net.seen) < 12800) {
                        box truth = { 0 };
                        truth.x = (i + .5) / l.w;
                        truth.y = (j + .5) / l.h;
                        truth.w = l.biases[2 * n];
                        truth.h = l.biases[2 * n + 1];
                        if (DOABS) {
                            truth.w = l.biases[2 * n] / l.w;
                            truth.h = l.biases[2 * n + 1] / l.h;
                        }
                        delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                    }
                }
            }
        }
        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box(state.truth + t * l.truth_size + b * l.truths);
            int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
            if (class_id >= l.classes) {
                printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                getchar();
                continue; // if label contains class_id more than number of classes in the cfg-file
            }

            if (!truth.x) break; // continue;
            float best_iou = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for (n = 0; n < l.n; ++n) {
                int index = size * (j * l.w * l.n + i * l.n + n) + b * l.outputs;
                box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                if (l.bias_match) {
                    pred.w = l.biases[2 * n];
                    pred.h = l.biases[2 * n + 1];
                    if (DOABS) {
                        pred.w = l.biases[2 * n] / l.w;
                        pred.h = l.biases[2 * n + 1] / l.h;
                    }
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_index = index;
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            float iou = delta_region_box(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
            if (iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 4];
            l.delta[best_index + 4] = l.object_scale * (1 - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            if (l.rescore) {
                l.delta[best_index + 4] = l.object_scale * (iou - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            }

            if (l.map) class_id = l.map[class_id];
            delta_region_class(l.output, l.delta, best_index + 5, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
#ifndef GPU
    flatten(l.delta, l.w * l.h, size * l.n, l.batch, 0);
#endif
    * (l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w * l.h * l.n * l.batch), recall / count, count);
}


void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, state.delta, 1);
}