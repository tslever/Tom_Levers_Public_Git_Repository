#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H


#include "darknet.h"


layer make_activation_layer(int batch, int inputs, ACTIVATION activation);
void forward_activation_layer(layer l, network_state state);
void backward_activation_layer(layer l, network_state state);


#endif