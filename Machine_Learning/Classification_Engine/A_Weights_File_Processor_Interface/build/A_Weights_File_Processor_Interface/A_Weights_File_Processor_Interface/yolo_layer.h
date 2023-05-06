#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H


#include "darknet.h"


layer make_yolo_layer(int batch, int w, int h, int n, int total, int* mask, int classes, int max_boxes);
void forward_yolo_layer(const layer l, network_state state);
void backward_yolo_layer(const layer l, network_state state);


#endif