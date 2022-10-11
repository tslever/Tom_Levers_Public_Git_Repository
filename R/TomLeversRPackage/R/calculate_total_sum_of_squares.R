calculate_total_sum_of_squares <- function(linear_model) {
    response_values <- linear_model$model[,1]
    sum_of_response_values <- sum(response_values)
    square_of_sum_of_response_values <- sum_of_response_values^2
    number_of_observations <- nobs(linear_model)
    total_sum_of_squares <- (t(response_values) %*% response_values) - (square_of_sum_of_response_values / number_of_observations)
    return(total_sum_of_squares)
}
