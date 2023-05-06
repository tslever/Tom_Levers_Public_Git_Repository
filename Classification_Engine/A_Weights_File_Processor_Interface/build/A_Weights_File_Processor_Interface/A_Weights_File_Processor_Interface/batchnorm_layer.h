#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H


#include "darknet.h"


layer make_batchnorm_layer(int batch, int w, int h, int c, int train);
void forward_batchnorm_layer(layer l, network_state state);
void backward_batchnorm_layer(layer l, network_state state);
void update_batchnorm_layer(layer l, int batch, float learning_rate, float momentum, float decay);


#endif