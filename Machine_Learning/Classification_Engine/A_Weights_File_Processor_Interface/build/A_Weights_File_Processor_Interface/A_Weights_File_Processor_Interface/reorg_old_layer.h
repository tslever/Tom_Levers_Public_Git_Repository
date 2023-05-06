#ifndef REORG_OLD_LAYER_H
#define REORG_OLD_LAYER_H


#include "darknet.h"


layer make_reorg_old_layer(int batch, int w, int h, int c, int stride, int reverse);
void forward_reorg_old_layer(const layer l, network_state state);
void backward_reorg_old_layer(const layer l, network_state state);


#endif