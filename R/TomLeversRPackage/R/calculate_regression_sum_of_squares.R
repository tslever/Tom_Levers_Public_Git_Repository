calculate_regression_sum_of_squares <- function(linear_model) {
    total_sum_of_squares <- calculate_total_sum_of_squares(linear_model)
    residual_sum_of_squares <- calculate_residual_sum_of_squares(linear_model)
    regression_sum_of_squares <- total_sum_of_squares - residual_sum_of_squares
    return(regression_sum_of_squares)
}
