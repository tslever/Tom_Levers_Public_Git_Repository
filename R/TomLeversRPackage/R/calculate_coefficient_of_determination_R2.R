calculate_coefficient_of_determination_R2 <- function(linear_model) {
    residual_sum_of_squares <- calculate_residual_sum_of_squares(linear_model)
    total_sum_of_squares <- calculate_total_sum_of_squares(linear_model)
    coefficient_of_determination_R2 <- 1 - (residual_sum_of_squares / total_sum_of_squares)
    return(coefficient_of_determination_R2)
}
