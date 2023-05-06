#ifndef RNN_LAYER_H
#define RNN_LAYER_H


#include "darknet.h"


layer make_rnn_layer(int batch, int inputs, int hidden, int outputs, int steps, ACTIVATION activation, int batch_normalize, int log);
void forward_rnn_layer(layer l, network_state state);
void backward_rnn_layer(layer l, network_state state);
void update_rnn_layer(layer l, int batch, float learning_rate, float momentum, float decay);


#endif