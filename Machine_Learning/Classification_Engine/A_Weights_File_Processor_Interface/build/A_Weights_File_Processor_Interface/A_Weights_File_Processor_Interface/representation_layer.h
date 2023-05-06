#ifndef REPRESENTATION_LAYER_H
#define REPRESENTATION_LAYER_H


#include "darknet.h"


layer make_implicit_layer(int batch, int index, float mean_init, float std_init, int filters, int atoms);
void forward_implicit_layer(const layer l, network_state state);
void backward_implicit_layer(const layer l, network_state state);
void update_implicit_layer(layer l, int batch, float learning_rate_init, float momentum, float decay);


#endif