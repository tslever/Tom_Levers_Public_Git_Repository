calculate_residual_mean_square <- function(linear_model) {
    residual_sum_of_squares <- calculate_residual_sum_of_squares(linear_model)
    number_of_observations <- nobs(linear_model)
    number_of_variables <- length(names(linear_model$coefficients))
    residual_mean_square <- residual_sum_of_squares / (number_of_observations - number_of_variables)
    return(residual_mean_square)
}
