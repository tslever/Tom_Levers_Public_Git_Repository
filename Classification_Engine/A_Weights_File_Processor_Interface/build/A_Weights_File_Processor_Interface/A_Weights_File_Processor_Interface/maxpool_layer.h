#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H


#include "darknet.h"


typedef layer maxpool_layer;


maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train);
void forward_local_avgpool_layer(const maxpool_layer l, network_state state);
void backward_local_avgpool_layer(const maxpool_layer l, network_state state);
void forward_maxpool_layer(const maxpool_layer l, network_state state);
void backward_maxpool_layer(const maxpool_layer l, network_state state);


#endif