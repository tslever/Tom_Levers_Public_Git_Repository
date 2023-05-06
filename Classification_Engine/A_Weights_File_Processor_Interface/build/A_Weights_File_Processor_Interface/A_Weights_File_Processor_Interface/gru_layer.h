#ifndef GRU_LAYER_H
#define GRU_LAYER_H


#include "darknet.h"


layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);
void forward_gru_layer(layer l, network_state state);
void backward_gru_layer(layer l, network_state state);
void update_gru_layer(layer l, int batch, float learning_rate, float momentum, float decay);


#endif