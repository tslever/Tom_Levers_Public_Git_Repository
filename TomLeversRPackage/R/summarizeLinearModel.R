#' @title summarizeLinearModel
#' @description Summarizes a linear model
#' @param linear_model The linear model for which to generate a summary
#' @return The summary for the linear model
#' @examples the_summary <- summarizeLinearModel(the_linear_model)
#' @import stringr
#' @export

summarizeLinearModel <- function(linear_model) {
    the_summary <- capture.output(summary(linear_model))
    the_regular_expression_for_a_decimal <- "-?\\d+(\\.\\d+)?"

    the_line_with_the_response_intercept <- the_summary[11]
    the_intercept <- str_extract(the_line_with_the_response_intercept, the_regular_expression_for_a_decimal)
    the_line_with_the_slope <- the_summary[12]
    the_slope <- str_extract(the_line_with_the_slope, the_regular_expression_for_a_decimal)
    the_line_with_the_linear_regression_equation <- paste("E(y | x) = B_0 + B_1 * x = ", the_intercept, " + ", the_slope, " * x", sep = "")
    the_summary <- append(the_summary, the_line_with_the_linear_regression_equation)

    the_line_with_coefficients_of_determination <- the_summary[17]
    multiple_and_adjusted_R_squared <- str_extract_all(the_line_with_coefficients_of_determination, the_regular_expression_for_a_decimal)[[1]]
    multiple_R_squared <- as.double(multiple_and_adjusted_R_squared[1])
    adjusted_R_squared <- as.double(multiple_and_adjusted_R_squared[2])
    multiple_R <- sqrt(multiple_R_squared)
    adjusted_R <- sqrt(adjusted_R_squared)
    the_line_with_the_correlation_coefficients <- paste("Multiple R:  ", multiple_R, "\tAdjusted R:  ", adjusted_R, sep = "")
    the_summary <- append(the_summary, the_line_with_the_correlation_coefficients)

    the_number_of_observations <- nobs(linear_model)
    the_line_with_the_number_of_observations <- paste("Number of observations: ", the_number_of_observations, sep = "")
    the_summary <- append(the_summary, the_line_with_the_number_of_observations)

    the_summary <- paste(the_summary, collapse = "\n")
    class(the_summary) <- "LinearModelSummary"
    return(the_summary)
}


#' @title print.linearModelSummary
#' @description Adds to the generic function print the LinearModelSummary method print.LinearModelSummary
#' @param the_summary The linear-model summary to print
#' @examples print(the_linear_model_summary)
#' @export

print.LinearModelSummary <- function(the_linear_model_summary) {
    cat(the_linear_model_summary)
}
