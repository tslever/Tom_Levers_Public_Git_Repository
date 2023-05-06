#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H


#include "darknet.h"


layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);
void forward_lstm_layer(layer l, network_state state);
void backward_lstm_layer(layer l, network_state state);
void update_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay);


#endif