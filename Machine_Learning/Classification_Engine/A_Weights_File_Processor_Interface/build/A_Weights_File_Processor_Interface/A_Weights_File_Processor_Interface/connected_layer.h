#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H


#include "darknet.h"


typedef layer connected_layer;

connected_layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize);
void forward_connected_layer(connected_layer layer, network_state state);
void backward_connected_layer(connected_layer layer, network_state state);
void update_connected_layer(connected_layer layer, int batch, float learning_rate, float momentum, float decay);
size_t get_connected_workspace_size(layer l);


#endif