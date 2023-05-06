#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H


#include "darknet.h"


typedef layer softmax_layer;
typedef layer contrastive_layer;


void softmax_tree(float* input, int batch, int inputs, float temp, tree* hierarchy, float* output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network_state state);
void backward_softmax_layer(const softmax_layer l, network_state state);
contrastive_layer make_contrastive_layer(int batch, int w, int h, int n, int classes, int inputs, layer* yolo_layer);
void forward_contrastive_layer(contrastive_layer l, network_state state);
void backward_contrastive_layer(contrastive_layer l, network_state net);


#endif