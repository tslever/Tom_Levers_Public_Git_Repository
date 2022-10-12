calculate_F_statistic <- function(linear_model) {
    regression_mean_square <- calculate_regression_mean_square(linear_model)
    residual_mean_square <- calculate_residual_mean_square(linear_model)
    F_statistic <- regression_mean_square / residual_mean_square
    return(F_statistic)
}
