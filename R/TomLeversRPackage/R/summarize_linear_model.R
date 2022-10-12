#' @title summarize_linear_model
#' @description Summarizes a linear model
#' @param linear_model The linear model to summarize
#' @return The summary for the linear model
#' @examples summary_output <- summarize_linear_model(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))

#' @export
summarize_linear_model <- function(linear_model) {
    the_summary <- summary(linear_model)
    summary_output <- capture.output(the_summary)

    coefficients <- linear_model$coefficients
    variables <- attr(linear_model$terms, "variables")
    number_of_variables <- length(variables)
    predictors <- attr(linear_model$terms, "term.labels")
    number_of_predictors <- length(predictors)
    substrings_of_generic_linear_regression_equation <- character(number_of_variables)
    substrings_of_generic_linear_regression_equation[1] <- paste("E(y | x) =\n    B_0", sep = "")
    for (i in 1:number_of_predictors) {
        substrings_of_generic_linear_regression_equation[i + 1] <- paste(" +\n    B_", predictors[i], " * ", predictors[i], sep = "")
    }
    generic_linear_regression_equation <- paste(substrings_of_generic_linear_regression_equation, sep = "", collapse = "")
    summary_output <- append(summary_output, generic_linear_regression_equation)

    substrings_of_linear_regression_equation <- character(number_of_variables)
    substrings_of_linear_regression_equation[1] <- paste("E(y | x) =\n    ", coefficients[1], sep = "")
    for (i in 1:number_of_predictors) {
     substrings_of_linear_regression_equation[i + 1] <- paste(" +\n    ", coefficients[i + 1], " * ", predictors[i], sep = "")
    }
    linear_regression_equation <- paste(substrings_of_linear_regression_equation, sep = "", collapse = "")
    summary_output <- append(summary_output, linear_regression_equation)

    number_of_observations <- nobs(linear_model)
    line_with_number_of_observations <- paste("Number of observations: ", number_of_observations, sep = "")
    summary_output <- append(summary_output, line_with_number_of_observations)

    residual_standard_error <- sigma(linear_model)
    estimated_variance_of_errors <- residual_standard_error ^ 2
    line_with_estimated_variance_of_errors <- paste("Estimated variance of errors: ", estimated_variance_of_errors, sep = "")
    summary_output <- append(summary_output, line_with_estimated_variance_of_errors)

    multiple_R_squared <- calculate_coefficient_of_determination_R2(linear_model)
    adjusted_R_squared <- calculate_adjusted_coefficient_of_determination_R2(linear_model)
    multiple_R <- sqrt(multiple_R_squared)
    #multiple_R <- cor(linear_model$model[,1], linear_model$model[,2], use = "complete.obs")
    adjusted_R <- sqrt(adjusted_R_squared)
    line_with_correlation_coefficients_R <- paste("Multiple R:  ", multiple_R, "\tAdjusted R:  ", adjusted_R, sep = "")
    summary_output <- append(summary_output, line_with_correlation_coefficients_R)

    critical_value_t <- calculate_critical_value_t(linear_model, 0.05)
    line_with_critical_value_t <- paste("Critical value t_{alpha/2, n-p}: ", critical_value_t, sep = "")
    summary_output <- append(summary_output, line_with_critical_value_t)

    summary_output <- paste(summary_output, collapse = "\n")
    class(summary_output) <- "linear_model_summary"
    return(summary_output)
}


#' @title print.linear_model_summary
#' @description Adds to the generic function print the linear_model_summary method print.linear_model_summary
#' @param summary_output The linear-model summary to print
#' @examples print(linear_model_summary)

#' @export
print.linear_model_summary <- function(linear_model_summary) {
    cat(linear_model_summary)
}
