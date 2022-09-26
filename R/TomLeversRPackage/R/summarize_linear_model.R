#' @title summarize_linear_model
#' @description Summarizes a linear model
#' @param linear_model The linear model to summarize
#' @return The summary for the linear model
#' @examples summary_output <- summarize_linear_model(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))
#' @import stringr

#' @export
summarize_linear_model <- function(linear_model) {
    the_summary <- summary(linear_model)
    summary_output <- capture.output(the_summary)

    response_intercept <- linear_model$coefficients[1]
    slope <- linear_model$coefficients[2]
    linear_regression_equation <- paste("E(y | x) = B_0 + B_1 * x = ", response_intercept, " + ", slope, " * x", sep = "")
    summary_output <- append(summary_output, linear_regression_equation)

    number_of_observations <- nobs(linear_model)
    line_with_number_of_observations <- paste("Number of observations: ", number_of_observations, sep = "")
    summary_output <- append(summary_output, line_with_number_of_observations)

    residual_standard_error <- sigma(linear_model)
    estimated_variance_of_errors <- residual_standard_error ^ 2
    line_with_estimated_variance_of_errors <- paste("Estimated variance of errors: ", estimated_variance_of_errors, sep = "")
    summary_output <- append(summary_output, line_with_estimated_variance_of_errors)

    multiple_R_squared <- the_summary$r.squared
    adjusted_R_squared <- the_summary$adj.r.squared
    multiple_R <- sqrt(multiple_R_squared)
    #multiple_R <- cor(linear_model$model[,1], linear_model$model[,2], use = "complete.obs")
    adjusted_R <- sqrt(adjusted_R_squared)
    line_with_correlation_coefficients_R <- paste("Multiple R:  ", multiple_R, "\tAdjusted R:  ", adjusted_R, sep = "")
    summary_output <- append(summary_output, line_with_correlation_coefficients_R)

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
