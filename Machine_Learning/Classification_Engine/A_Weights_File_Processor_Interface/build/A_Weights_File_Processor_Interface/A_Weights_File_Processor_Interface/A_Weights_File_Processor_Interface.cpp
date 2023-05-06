#include "pch.h" // Must be the first inclusion
#include "A_Weights_File_Processor_Interface.h"
#include <iostream>
#include "parser.h"
#include "network.h"


network DNLIB_parse_network_cfg_custom(char* filename, int batch, int time_steps) {

	//printf("Testing DNLIB_parse_network_cfg_custom.\n");

	//network* the_pointer_to_the_network = (network*)malloc(sizeof(network));
	//return *the_pointer_to_the_network;

	return parse_network_cfg_custom(filename, batch, time_steps);

}


void DNLIB_load_weights(network* net, char* filename) {

	//printf("Testing DNLIB_load_weights.\n");

	load_weights(net, filename);

}


network DNLIB_provides_a_neural_network_with_loaded_weights_based_on(char* the_name_of_the_network_configuration_file, char* the_name_of_the_weights_file) {

	network the_neural_network = parse_network_cfg_custom(the_name_of_the_network_configuration_file, 1, 0);

	load_weights(&the_neural_network, the_name_of_the_weights_file);

	set_batch_network(&the_neural_network, 1);

	srand(2222222);

	fuse_conv_batchnorm(the_neural_network);

	calculate_binary_weights(the_neural_network);

	return the_neural_network;

}