calculate_adjusted_coefficient_of_determination_R2 <- function(linear_model) {
    residual_mean_square <- calculate_residual_mean_square(linear_model)
    total_mean_square <- calculate_total_mean_square(linear_model)
    adjusted_coefficient_of_determination_R2 <- 1 - (residual_mean_square / total_mean_square)
    return(adjusted_coefficient_of_determination_R2)
}
