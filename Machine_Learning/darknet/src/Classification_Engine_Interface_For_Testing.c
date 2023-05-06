#include "Classification_Engine_Interface_For_Testing.h"
#include "darknet.h"


float* DNLIB_network_predict_for_testing(network net, float* input) {

    printf("Testing DNLIB_network_predict_for_testing.\n");

    return NULL;

}


void DNLIB_hierarchy_predictions_for_testing(float* predictions, int n, tree* hier, int only_leaves) {

    printf("Testing DNLIB_hierarchy_predictions_for_testing.\n");

}


void DNLIB_top_k_for_testing(float* a, int n, int k, int* index) {

    printf("Testing DNLIB_top_k_for_testing.\n");

}