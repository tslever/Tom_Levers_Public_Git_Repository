#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H


#include "darknet.h"


typedef layer convolutional_layer;

convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer* share_layer, int assisted_excitation, int deform, int train);
void forward_convolutional_layer(const convolutional_layer layer, network_state state);
void backward_convolutional_layer(convolutional_layer layer, network_state state);
int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);
void update_convolutional_layer(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);
size_t get_convolutional_workspace_size(layer l);
void binarize_weights(float* weights, int n, int size, float* binary);
void swap_binary(convolutional_layer* l);
void assisted_excitation_forward(convolutional_layer l, network_state state);
void backward_bias(float* bias_updates, float* delta, int batch, int n, int size);
size_t get_convolutional_workspace_size(layer l);
void free_convolutional_batchnorm(convolutional_layer* l);
void binary_align_weights(convolutional_layer* l);


#endif