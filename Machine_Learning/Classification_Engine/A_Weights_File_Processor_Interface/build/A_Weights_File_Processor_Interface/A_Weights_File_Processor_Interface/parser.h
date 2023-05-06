#ifndef PARSER_H
#define PARSER_H


#include "darknet.h"


void load_weights(network* net, char* filename);
void load_weights_upto(network* net, char* filename, int cutoff);
void load_convolutional_weights(layer l, FILE* fp);
void transpose_matrix(float* a, int rows, int cols);
void load_shortcut_weights(layer l, FILE* fp);
void load_implicit_weights(layer l, FILE* fp);
void load_connected_weights(layer l, FILE* fp, int transpose);
void load_batchnorm_weights(layer l, FILE* fp);
network parse_network_cfg_custom(char* filename, int batch, int time_steps);


#endif