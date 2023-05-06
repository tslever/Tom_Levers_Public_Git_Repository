#ifndef CONV_LSTM_LAYER_H
#define CONV_LSTM_LAYER_H


#include "darknet.h"


layer make_conv_lstm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int peephole, int xnor, int bottleneck, int train);
void forward_conv_lstm_layer(layer l, network_state state);
void backward_conv_lstm_layer(layer l, network_state state);
void update_conv_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay);
layer make_history_layer(int batch, int h, int w, int c, int history_size, int steps, int train);
void forward_history_layer(layer l, network_state state);
void backward_history_layer(layer l, network_state state);


#endif