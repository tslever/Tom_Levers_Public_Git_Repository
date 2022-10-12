calculate_total_mean_square <- function(linear_model) {
    total_sum_of_squares <- calculate_total_sum_of_squares(linear_model)
    number_of_observations <- nobs(linear_model)
    total_mean_square <- total_sum_of_squares / (number_of_observations - 1)
    return(total_mean_square)
}
