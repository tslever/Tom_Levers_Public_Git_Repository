#ifndef REORG_LAYER_H
#define REORG_LAYER_H


#include "darknet.h"


layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse);
void forward_reorg_layer(const layer l, network_state state);
void backward_reorg_layer(const layer l, network_state state);


#endif
