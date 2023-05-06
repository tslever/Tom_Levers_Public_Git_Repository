#include "Classification_Engine_Interface.h"
#include "darknet.h"


float* DNLIB_network_predict(network net, float* input) {

    return network_predict(net, input);

}


void DNLIB_hierarchy_predictions(float* predictions, int n, tree* hier, int only_leaves) {

    hierarchy_predictions(predictions, n, hier, only_leaves);

}


void DNLIB_top_k(float* a, int n, int k, int* index) {

    top_k(a, n, k, index);

}