#' @title calculate_residual_mean_square
#' @description Calculates the residual mean square of a multiple linear model
#' @param linear_model The linear model for which to calculate the residual mean square
#' @return The residual mean square
#' @examples residual_mean_square <- calculate_residual_mean_square(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))

#' @export
calculate_residual_mean_square <- function(linear_model) {
    residual_sum_of_squares <- calculate_residual_sum_of_squares(linear_model)
    number_of_observations <- nobs(linear_model)
    number_of_variables <- length(names(linear_model$coefficients))
    residual_mean_square <- residual_sum_of_squares / (number_of_observations - number_of_variables)
    return(residual_mean_square)
}
