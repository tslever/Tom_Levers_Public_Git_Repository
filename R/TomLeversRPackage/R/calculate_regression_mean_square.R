calculate_regression_mean_square <- function(linear_model) {
    regression_sum_of_squares <- calculate_regression_sum_of_squares(linear_model)
    regression_degrees_of_freedom <- calculate_regression_degrees_of_freedom(linear_model)
    regression_mean_square <- regression_sum_of_squares / regression_degrees_of_freedom
    return(regression_mean_square)
}
