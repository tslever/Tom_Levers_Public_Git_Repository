#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H


#include "darknet.h"


typedef layer route_layer;


route_layer make_route_layer(int batch, int n, int* input_layers, int* input_size, int groups, int group_id);
void forward_route_layer(const route_layer l, network_state state);
void backward_route_layer(const route_layer l, network_state state);


#endif