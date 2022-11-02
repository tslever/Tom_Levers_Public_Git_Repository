#' @title calculate_PRESS
#' @description Calculates the Predicted Residual Error Sum of Squares for a linear model
#' @param linear_model The linear model for which to calculate PRESS
#' @return PRESS
#' @examples PRESS <- calculate_PRESS(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))

#' @export
calculate_PRESS <- function(linear_model) {
    vector_of_residuals <- linear_model$residuals
    list_of_quantities_for_diagnostics_for_checking_quality_of_regression_fits <- lm.influence(linear_model)
    diagonal_of_hat_matrix <- list_of_quantities_for_diagnostics_for_checking_quality_of_regression_fits$hat
    PRESS <- sum((vector_of_residuals / (1 - diagonal_of_hat_matrix))^2)
    return(PRESS)
}
