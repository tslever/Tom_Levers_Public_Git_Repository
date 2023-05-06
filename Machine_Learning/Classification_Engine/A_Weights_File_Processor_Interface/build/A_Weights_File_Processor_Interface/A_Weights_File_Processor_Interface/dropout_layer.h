#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H


#include "darknet.h"


typedef layer dropout_layer;


dropout_layer make_dropout_layer(int batch, int inputs, float probability, int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c);
void forward_dropout_layer(dropout_layer l, network_state state);
void backward_dropout_layer(dropout_layer l, network_state state);


#endif