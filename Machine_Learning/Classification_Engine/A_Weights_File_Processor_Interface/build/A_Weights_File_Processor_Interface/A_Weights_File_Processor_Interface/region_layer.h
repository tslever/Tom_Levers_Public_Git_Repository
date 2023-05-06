#ifndef REGION_LAYER_H
#define REGION_LAYER_H


#include "darknet.h"


typedef layer region_layer;


region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes);
void forward_region_layer(const region_layer l, network_state state);
void backward_region_layer(const region_layer l, network_state state);


#endif