
import numpy as np
import zipfile
import os
import cv2


# First cell.
def Prepare_Training_and_Testing_Datasets():

    with zipfile.ZipFile("Quercus_Dataset--NumPy_NNs.zip", "r") as z:
        z.extractall("./")

    path_to_training_dataset = "./Quercus_Dataset/Training_Dataset/"
    path_to_testing_dataset = "./Quercus_Dataset/Testing_Dataset/"

    def Prepare_Dataset(path_to_dataset):

        print("\nPreparing dataset at " + chr(34) + path_to_dataset + chr(34) + ".\n")

        list_of_paths = os.listdir(path_to_dataset)

        number_of_images = int(len(list_of_paths) / 2)

        width_of_image, height_of_image, number_of_channels = (416, 416, 3)

        number_of_values_in_image = width_of_image * height_of_image * number_of_channels

        matrix_of_flattened_normalized_images = np.zeros( (number_of_values_in_image, number_of_images) )

        for i in range(0, number_of_images):

            image = cv2.imread( path_to_dataset + list_of_paths[2*i] )

            normalized_resized_image = cv2.resize( image, (width_of_image, height_of_image) ) / 255

            area = width_of_image * number_of_channels
            
            flattened_horizontal_slice_of_image = np.zeros( (area, 1) )

            for j in range(0, height_of_image):

                for k in range(0, width_of_image):

                    flattened_horizontal_slice_of_image[k * number_of_channels : (k+1) * number_of_channels, 0] = \
                        normalized_resized_image[j][k]

                    matrix_of_flattened_normalized_images[j * area : (j+1) * area, i] = flattened_horizontal_slice_of_image[:, 0]

            print("Added Image " + str(i+1) + " to matrix of flattened, normalized images.")

        vector_of_ground_truth_values = np.ones( (1, number_of_images) )

        return matrix_of_flattened_normalized_images, vector_of_ground_truth_values

    matrix_of_flat_normalized_training_images, vect_of_ground_truth_values_for_training_dataset = \
        Prepare_Dataset(path_to_training_dataset)

    matrix_of_flat_normalized_testing_images, vect_of_ground_truth_values_for_testing_dataset = \
        Prepare_Dataset(path_to_testing_dataset)

    return (matrix_of_flat_normalized_training_images, vect_of_ground_truth_values_for_training_dataset, \
            matrix_of_flat_normalized_testing_images, vect_of_ground_truth_values_for_testing_dataset)

matrix_of_flattened_normalized_training_images, vector_of_ground_truth_values_for_training_dataset, \
matrix_of_flattened_normalized_testing_images, vector_of_ground_truth_values_for_testing_dataset = \
    Prepare_Training_and_Testing_Datasets()


# Second cell.
vector_of_weights = np.zeros( (matrix_of_flattened_normalized_training_images.shape[0], 1) )
bias = 0
offset_to_prevent_division_by_zero = 0
scale_for_normalized_output_of_linear_function = 0
offset_for_normalized_output_of_linear_function = 0

learning_rate = 0
number_of_epochs = 0


# Third cell.
def Initialize_Parameters_and_Hyperparameters(matrix_of_flat_normalized_training_images):

    ########################
    # Initialize parameters.
    ########################
    
    number_of_values_in_image = matrix_of_flat_normalized_training_images.shape[0]
    initial_vector_of_weights = np.random.randn( number_of_values_in_image, 1 ) * np.sqrt(1 / number_of_values_in_image)

    initial_bias = 0
    
    initial_output_of_linear_function = \
        np.dot(initial_vector_of_weights.T, matrix_of_flat_normalized_training_images) + initial_bias
    
    mean_of_initial_output_of_linear_function = np.mean(initial_output_of_linear_function)
    
    variance_of_initial_output_of_linear_function = \
        np.mean( (initial_output_of_linear_function - mean_of_initial_output_of_linear_function)**2 )
    
    offset_to_prevent_div_by_zero = 0.000001
    
    initial_scale_for_normalized_output_of_linear_function = \
        np.sqrt(variance_of_initial_output_of_linear_function + offset_to_prevent_div_by_zero)
    
    initial_offset_for_normalized_output_of_linear_function = mean_of_initial_output_of_linear_function
    
    
    #############################
    # Initialize hyperparameters.
    #############################
    
    # Initialize the learning rate (a hyperparameter) as 0.1.
    learn_rate = 0.1

    # Initialize the number of epochs (a hyperparameters) as 1000.
    num_of_epochs = 100
    
    ################################################
    # Return initialized training dataset variables,
    # parameters, and hyperparameters.
    ################################################
    
    return (initial_vector_of_weights, \
            initial_bias, \
            offset_to_prevent_div_by_zero, \
            initial_scale_for_normalized_output_of_linear_function, \
            initial_offset_for_normalized_output_of_linear_function, \
            learn_rate, \
            num_of_epochs)


vector_of_weights, \
bias, \
offset_to_prevent_division_by_zero, \
scale_for_normalized_output_of_linear_function, \
offset_for_normalized_output_of_linear_function, \
learning_rate, \
number_of_epochs = \
Initialize_Parameters_and_Hyperparameters(matrix_of_flattened_normalized_training_images)


# Fourth cell.
def Train_Neural_Network(num_of_epochs, \
                         vect_of_weights, \
                         matrix_of_flat_normalized_training_images, \
                         b, \
                         offset_to_prevent_div_by_zero, \
                         scale, \
                         offset_for_norm_output_of_linear_function, \
                         vect_of_ground_truth_values_for_training_dataset, \
                         learn_rate):
    
    
    number_of_images = matrix_of_flat_normalized_training_images.shape[1]
        
    vector_of_ones = np.ones( (1, number_of_images) )
    
    for i in range(0, num_of_epochs):
    
        ####################
        # Forward propagate.
        ####################
    
        output_of_linear_function = np.dot(vect_of_weights.T, matrix_of_flat_normalized_training_images) + b
        
        mean_of_output_of_linear_function = np.mean(output_of_linear_function)
        
        variance_of_output_of_linear_function = np.mean( (output_of_linear_function - mean_of_output_of_linear_function)**2 )
        
        sqrt_of_sum_of_variance_of_output_of_linear_function_and_offset_to_prevent_div_by_zero = \
            np.sqrt(variance_of_output_of_linear_function + offset_to_prevent_div_by_zero)
        
        normalized_output_of_linear_function = \
            (output_of_linear_function - mean_of_output_of_linear_function) / \
            sqrt_of_sum_of_variance_of_output_of_linear_function_and_offset_to_prevent_div_by_zero
        
        scaled_and_offset_normalized_output_of_linear_function = \
            scale * normalized_output_of_linear_function + offset_for_norm_output_of_linear_function
        
        vector_of_predicted_values = 1 / (1 + np.exp(-scaled_and_offset_normalized_output_of_linear_function))
    
        cost = -np.mean( vect_of_ground_truth_values_for_training_dataset * np.log(vector_of_predicted_values) + \
                         (1-vect_of_ground_truth_values_for_training_dataset) * np.log(1-vector_of_predicted_values) )

        print(cost)

        ##################################################
        
        vector_of_derivatives_of_cost_wrt_predicted_value = \
            (vector_of_predicted_values - vect_of_ground_truth_values_for_training_dataset) / \
            (number_of_images * vector_of_predicted_values * (1 - vector_of_predicted_values))
        
        vector_of_derivatives_of_predicted_value_wrt_SON_output_of_linear_function = \
            vector_of_predicted_values * (1 - vector_of_predicted_values)
        
        vector_of_derivatives_of_cost_wrt_SON_output_of_linear_function = \
            vector_of_derivatives_of_cost_wrt_predicted_value * \
            vector_of_derivatives_of_predicted_value_wrt_SON_output_of_linear_function
        
        vector_of_derivatives_of_SON_output_of_linear_function_wrt_normalized_output_of_linear_function = \
            vector_of_ones * scale
        
        difference_between_output_of_linear_function_and_mean_of_output_of_linear_function = \
            output_of_linear_function - mean_of_output_of_linear_function
        
        mean_of_difference = np.mean( difference_between_output_of_linear_function_and_mean_of_output_of_linear_function )
        
        vector_of_derivatives_of_normalized_output_of_linear_function_wrt_output_of_linear_function = \
            mean_of_difference / sqrt_of_sum_of_variance_of_output_of_linear_function_and_offset_to_prevent_div_by_zero**3 * \
            difference_between_output_of_linear_function_and_mean_of_output_of_linear_function + \
            (number_of_images - 1) / (number_of_images * sqrt_of_sum_of_variance_of_output_of_linear_function_and_offset_to_prevent_div_by_zero)
        
        vector_of_derivatives_of_cost_wrt_output_of_linear_function = \
            vector_of_derivatives_of_cost_wrt_SON_output_of_linear_function * \
            vector_of_derivatives_of_SON_output_of_linear_function_wrt_normalized_output_of_linear_function * \
            vector_of_derivatives_of_normalized_output_of_linear_function_wrt_output_of_linear_function
        
        vector_of_derivatives_of_cost_wrt_weight = \
            np.sum( vector_of_derivatives_of_cost_wrt_output_of_linear_function * \
                    matrix_of_flat_normalized_training_images, axis = 1 )
        
        vect_of_weights = vect_of_weights - learn_rate * np.expand_dims(vector_of_derivatives_of_cost_wrt_weight, axis=1)
        
        ##################################################
        
        derivative_of_cost_wrt_bias = np.sum( vector_of_derivatives_of_cost_wrt_output_of_linear_function )
        
        b = b - learn_rate * derivative_of_cost_wrt_bias                
        
        ##################################################

        vector_of_derivatives_of_SON_output_of_linear_function_wrt_scale = normalized_output_of_linear_function
        
        derivative_of_cost_wrt_scale = \
            np.sum( vector_of_derivatives_of_cost_wrt_SON_output_of_linear_function * \
                    vector_of_derivatives_of_SON_output_of_linear_function_wrt_scale )
        
        scale = scale - learning_rate * derivative_of_cost_wrt_scale

        ##################################################

        #vector_of_derivatives_of_SON_output_of_linear_function_wrt_offset = 1
        
        #derivative_of_cost_wrt_offset_for_normalized_output_of_linear_function = \
        #    np.sum( vector_of_derivatives_of_cost_wrt_SON_output_of_linear_function * \
        #            vector_of_derivatives_of_SON_output_of_linear_function_wrt_offset )
        
        derivative_of_cost_wrt_offset_for_normalized_output_of_linear_function = \
            np.sum( vector_of_derivatives_of_cost_wrt_SON_output_of_linear_function )
        
        offset_for_norm_output_of_linear_function = \
            offset_for_norm_output_of_linear_function - \
            learning_rate * derivative_of_cost_wrt_offset_for_normalized_output_of_linear_function

        ##################################################
    
    
    return (vect_of_weights, \
            b, \
            scale, \
            offset_for_norm_output_of_linear_function)
        

vector_of_weights, \
bias, \
scale_for_normalized_output_of_linear_function, \
offset_for_normalized_output_of_linear_function = \
Train_Neural_Network(number_of_epochs, \
                     vector_of_weights, \
                     matrix_of_flattened_normalized_training_images, \
                     bias, \
                     offset_to_prevent_division_by_zero, \
                     scale_for_normalized_output_of_linear_function, \
                     offset_for_normalized_output_of_linear_function, \
                     vector_of_ground_truth_values_for_training_dataset, \
                     learning_rate)


# Fifth cell.
output_of_linear_function = \
    np.dot(vector_of_weights.T, matrix_of_flattened_normalized_testing_images) + bias

mean_of_output_of_linear_function = np.mean(output_of_linear_function)

variance_of_output_of_linear_function = np.mean( (output_of_linear_function - mean_of_output_of_linear_function)**2 )

normalized_output_of_linear_function = \
    (output_of_linear_function - mean_of_output_of_linear_function) / \
    np.sqrt(variance_of_output_of_linear_function + offset_to_prevent_division_by_zero)

scaled_and_offset_normalized_output_of_linear_function = \
    scale_for_normalized_output_of_linear_function * normalized_output_of_linear_function + \
    offset_for_normalized_output_of_linear_function

vector_of_predicted_values = 1 / (1 + np.exp(-scaled_and_offset_normalized_output_of_linear_function))

print(vector_of_predicted_values)