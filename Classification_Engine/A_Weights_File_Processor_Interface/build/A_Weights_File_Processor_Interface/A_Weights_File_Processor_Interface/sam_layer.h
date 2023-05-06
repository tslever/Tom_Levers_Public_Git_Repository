#ifndef SAM_CHANNELS_LAYER_H
#define SAM_CHANNELS_LAYER_H


#include "darknet.h"


layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_sam_layer(const layer l, network_state state);
void backward_sam_layer(const layer l, network_state state);


#endif