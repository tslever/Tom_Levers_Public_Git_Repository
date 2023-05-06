#include <iostream>


#include "A_Weights_File_Processor_Interface.h"
#include "An_Image_Processor_Interface.h"
#include "A_Classification_Engine_Interface.h"


// TODO: Add C/C++ project property changes to VC++ properties.

// fopen_s at application level for reading NN CFG, pass FILE* or input stream or
// array of strings to DNLIB_provides_a_neural... Ideally, input stream.

// Standardize styling.

// cv::Mat comes into DNLIB_provides_a_pointer_to_the_data_from... in an overload.

// darknet.h: Move to a common location for all subsystems.

// Making sure that the application doesn't change, by wrapping the network type in a 
// custom type so that the application doesn't depend directly on darknet.
// e.g., Toms_Network in Image_Binary_Classifier.h

// Target: Implementor of main program that uses customized image processing for image binary classification.
// System: Customized image processing for image binary classification.
//     Subsystem 1: Training for a specific classification purpose, which outputs a Binary Classification Rules File (e.g., a weights file in darknet).
//     Subsystem 2: Use the Binary Classification Rules File to classify images.
//         Subsystem 2.1: Application that provides the Binary Classification Rules File and image(s) to Subsystem 2.2: The Image Binary Classifier.
//         Subsystem 2.2: The Image Binary Classifier: A general image binary classifier (This is currently implemented as three DLL's, and could be implemented using students, or a single DLL).
// Create system diagrams.

// Named argument associations for default approved


int main(int argc, char* argv[])
{
    char* The_Name_Of_The_Image_File = argv[1];
    char* The_Name_Of_The_Binary_Classification_Rules_File = argv[2];
    char* The_Name_Of_Network_Configuration_File = (char*)"PATH_TO_NETWORK_CONFIGURATION_FILE";
    if (argc > 2) {
        The_Name_Of_Network_Configuration_File = argv[3];
    }

    network the_neural_network_with_loaded_weights =
        DNLIB_provides_a_neural_network_with_loaded_weights_based_on(
            The_Name_Of_Network_Configuration_File, The_Name_Of_The_Binary_Classification_Rules_File
        );

    //float* the_pointer_to_the_data_from_a_loaded_resized_and_cropped_image =
    //    a_pointer_to_the_data_from_a_loaded_resized_and_cropped_image_based_on()
    float* the_pointer_to_the_data_from_a_loaded_resized_and_cropped_image =
        DNLIB_provides_a_pointer_to_the_data_from_a_loaded_resized_and_cropped_image_based_on(
            The_Name_Of_The_Image_File, the_neural_network_with_loaded_weights
        );


    DNLIB_network_predict(the_neural_network_with_loaded_weights, NULL);
    DNLIB_hierarchy_predictions(NULL, 0, NULL, 0);
    DNLIB_top_k(NULL, 0, 0, NULL);

}