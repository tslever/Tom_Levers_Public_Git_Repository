#' @title split_data_set_into_training_and_testing_sets
#' @description Splits data set into training and testing sets
#' @param data_frame The data set to split into training and testing sets
#' @param proportion_of_training_data The proportion of training data
#' @return split_data The split data
#' @examples split_data <- split_data_set_into_training_and_testing_sets(data_frame = Default, proportion_of_training_data = 0.5)

#' @export
split_data_set_into_training_and_testing_data <- function(data_frame, proportion_of_training_data) {
 number_of_observations <- nrow(data_frame)
 vector_of_random_indices <- sample(1:number_of_observations)
 shuffled_data_frame <- data_frame[vector_of_random_indices, ]
 number_of_training_data <- ceiling(number_of_observations * proportion_of_training_data)
 training_data <- shuffled_data_frame[1:number_of_training_data, ]
 testing_data <- shuffled_data_frame[(number_of_training_data + 1) : number_of_observations, ]
 split_data <- list(training_data = training_data, testing_data = testing_data)
 class(split_data) <- "split_data"
 return(split_data)
}
