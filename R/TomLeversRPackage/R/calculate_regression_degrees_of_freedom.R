#' @export
calculate_regression_degrees_of_freedom <- function(linear_model) {
    number_of_variables <- length(names(linear_model$coefficients))
    number_of_predictors <- number_of_variables - 1
    regression_degrees_of_freedom <- number_of_predictors
    return(regression_degrees_of_freedom)
}
