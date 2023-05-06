#ifndef COST_LAYER_H
#define COST_LAYER_H


#include "darknet.h"


typedef layer cost_layer;

COST_TYPE get_cost_type(char* s);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale);
void forward_cost_layer(const cost_layer l, network_state state);
void backward_cost_layer(const cost_layer l, network_state state);


#endif