#ifndef SCALE_CHANNELS_LAYER_H
#define SCALE_CHANNELS_LAYER_H


#include "darknet.h"


layer make_scale_channels_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int scale_wh);
void forward_scale_channels_layer(const layer l, network_state state);
void backward_scale_channels_layer(const layer l, network_state state);


#endif