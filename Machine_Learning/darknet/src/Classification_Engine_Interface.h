#ifndef CLASSIFICATION_ENGINE_INTERFACE_H
#define CLASSIFICATION_ENGINE_INTERFACE_H


#include "darknet.h"


float* DNLIB_network_predict(network net, float* input);
void DNLIB_hierarchy_predictions(float* predictions, int n, tree* hier, int only_leaves);
void DNLIB_top_k(float* a, int n, int k, int* index);


#endif