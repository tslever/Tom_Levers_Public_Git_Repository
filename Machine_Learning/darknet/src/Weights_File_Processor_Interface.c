#include "darknet.h"
#include "Weights_File_Processor_Interface.h"


void DNLIB_load_weights(network* net, char* filename) {

    load_weights(net, filename);

}