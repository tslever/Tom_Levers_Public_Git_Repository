#' @title calculate_mean_squared_error
#' @description Calculates mean squared error
#' @param model The model
#' @return mean_squared_error The mean squared error
#' @examples mean_squared_error <- calculate_mean_squared_error(model)

#' @export
calculate_mean_squared_error <- function(vector_of_predicted_response_values, vector_of_actual_response_values) {
 #vector_of_residuals <- model$residuals
 vector_of_residuals <- vector_of_predicted_response_values - vector_of_actual_response_values
 vector_of_squared_residuals <- vector_of_residuals^2
 sum_of_squared_residuals <- sum(vector_of_squared_residuals)
 number_of_residuals <- length(vector_of_residuals)
 mean_squared_error <- sum_of_squared_residuals / number_of_residuals
 return(mean_squared_error)
}
