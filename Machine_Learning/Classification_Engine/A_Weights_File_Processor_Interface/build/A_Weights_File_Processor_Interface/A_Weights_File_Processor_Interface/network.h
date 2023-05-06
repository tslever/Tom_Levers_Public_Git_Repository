#ifndef NETWORK_H
#define NETWORK_H


#include "darknet.h"


int get_current_batch(network net);
network make_network(int n);
int get_network_output_size(network net);
float* get_network_output(network net);
int64_t get_current_iteration(network net);
void set_batch_network(network* net, int b);


#endif