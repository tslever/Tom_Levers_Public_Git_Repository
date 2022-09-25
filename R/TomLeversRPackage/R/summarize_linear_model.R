#' @title summarize_linear_model
#' @description Summarizes a linear model
#' @param linear_model The linear model to summarize
#' @return The summary for the linear model
#' @examples the_summary <- summarize_linear_model(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))
#' @import stringr

#' @export
summarize_linear_model <- function(linear_model) {
    the_summary <- capture.output(summary(linear_model))
    regular_expression_for_number <- get_regular_expression_for_number()

    line_with_response_intercept <- the_summary[11]
    response_intercept <- str_extract(line_with_response_intercept, regular_expression_for_number)
    line_with_slope <- the_summary[12]
    slope <- str_extract(line_with_slope, regular_expression_for_number)
    linear_regression_equation <- paste("E(y | x) = B_0 + B_1 * x = ", response_intercept, " + ", slope, " * x", sep = "")
    the_summary <- append(the_summary, linear_regression_equation)

    number_of_observations <- nobs(linear_model)
    line_with_number_of_observations <- paste("Number of observations: ", number_of_observations, sep = "")
    the_summary <- append(the_summary, line_with_number_of_observations)

    line_with_residual_standard_error <- the_summary[16]
    residual_standard_error <- as.double(str_extract(line_with_residual_standard_error, regular_expression_for_number))
    estimated_variance_of_errors <- residual_standard_error ^ 2
    line_with_estimated_variance_of_errors <- paste("Estimated variance of errors: ", estimated_variance_of_errors, sep = "")
    the_summary <- append(the_summary, line_with_estimated_variance_of_errors)

    line_with_coefficients_of_determination_R2 <- the_summary[17]
    multiple_and_adjusted_R_squared <- str_extract_all(line_with_coefficients_of_determination_R2, regular_expression_for_number)[[1]]
    multiple_R_squared <- as.double(multiple_and_adjusted_R_squared[1])
    adjusted_R_squared <- as.double(multiple_and_adjusted_R_squared[2])
    #multiple_R <- sqrt(multiple_R_squared)
    multiple_R <- cor(linear_model$model[,1], linear_model$model[,2], use = "complete.obs")
    adjusted_R <- sqrt(adjusted_R_squared)
    line_with_correlation_coefficients_R <- paste("Multiple R:  ", multiple_R, "\tAdjusted R:  ", adjusted_R, sep = "")
    the_summary <- append(the_summary, line_with_correlation_coefficients_R)

    the_summary <- paste(the_summary, collapse = "\n")
    class(the_summary) <- "linear_model_summary"
    return(the_summary)
}


#' @title print.linear_model_summary
#' @description Adds to the generic function print the linear_model_summary method print.linear_model_summary
#' @param the_summary The linear-model summary to print
#' @examples print(linear_model_summary)
#' @export

print.linear_model_summary <- function(linear_model_summary) {
    cat(linear_model_summary)
}
