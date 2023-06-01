calculate_residual_sum_of_squares <- function(linear_model) {
    residuals <- linear_model$residuals
    residual_sum_of_squares <- t(residuals) %*% residuals
    residual_sum_of_squares <- residual_sum_of_squares[1, 1]
    return(residual_sum_of_squares)
}
