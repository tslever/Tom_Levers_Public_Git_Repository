#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H


#include "darknet.h"


typedef layer detection_layer;
detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);


#endif