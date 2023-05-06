#ifndef LOCAL_LAYER
#define LOCAL_LAYER


#include "darknet.h"


typedef layer local_layer;

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);
void forward_local_layer(const local_layer layer, network_state state);
void backward_local_layer(local_layer layer, network_state state);
void update_local_layer(local_layer layer, int batch, float learning_rate, float momentum, float decay);


#endif