#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H


#include "darknet.h"


layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void forward_normalization_layer(const layer layer, network_state state);
void backward_normalization_layer(const layer layer, network_state state);


#endif