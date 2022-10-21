#' @title analyze_variance_for_one_linear_model
#' @description Analyzes the variance of a linear model
#' @param linear_model The linear model to analyze
#' @return The analysis of variance for the linear model
#' @examples analysis_of_variance <- analyze_variance(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))

#' @export
analyze_variance_for_one_linear_model <- function(linear_model) {
    analysis <- capture.output(anova(linear_model))

    regression_degrees_of_freedom <- calculate_regression_degrees_of_freedom(linear_model)
    regression_sum_of_squares <- calculate_regression_sum_of_squares(linear_model)
    regression_mean_square <- calculate_regression_mean_square(linear_model)
    line_with_regression_degrees_of_freedom_sum_of_squares_and_mean_square <- paste("DFR: ", regression_degrees_of_freedom, ", SSR: ", regression_sum_of_squares, ", MSR: ", regression_mean_square, sep = "")
    analysis <- append(analysis, line_with_regression_degrees_of_freedom_sum_of_squares_and_mean_square)

    F_statistic <- calculate_test_statistic_F(linear_model)
    significance_level <- 0.05
    residual_degrees_of_freedom <- calculate_residual_degrees_of_freedom(linear_model)
    critical_F_value <- calculate_critical_value_Fc(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom)
    line_with_F_statistic_and_critical_F_value <- paste("F0: ", F_statistic, ", F(alpha = ", significance_level, ", DFR = ", regression_degrees_of_freedom, ", DFRes = ", residual_degrees_of_freedom, "): ", critical_F_value, sep = "")
    analysis <- append(analysis, line_with_F_statistic_and_critical_F_value)

    probability <- calculate_p_value_for_all_MLR_coefficients(linear_model)
    line_with_probability <- paste("p: ", probability, sep = "")
    analysis <- append(analysis, line_with_probability)

    total_degrees_of_freedom <- calculate_total_degrees_of_freedom(linear_model)
    total_sum_of_squares <- calculate_total_sum_of_squares(linear_model)
    line_with_totals <- paste("DFT: ", total_degrees_of_freedom, ", SST: ", total_sum_of_squares, sep = "")
    analysis <- append(analysis, line_with_totals)

    coefficient_of_determination_R2 <- calculate_coefficient_of_determination_R2(linear_model)
    adjusted_coefficient_of_determination_R2 <- calculate_adjusted_coefficient_of_determination_R2(linear_model)
    line_with_adjusted_coefficient_of_determination_R2 <- paste("R2: ", coefficient_of_determination_R2, ", Adjusted R2: ", adjusted_coefficient_of_determination_R2, sep = "")
    analysis <- append(analysis, line_with_adjusted_coefficient_of_determination_R2)

    number_of_observations <- nobs(linear_model)
    line_with_number_of_observations <- paste("Number of observations: ", number_of_observations, sep = "")
    analysis <- append(analysis, line_with_number_of_observations)

    analysis <- paste(analysis, collapse = "\n")
    class(analysis) <- "analysis_of_variance"
    return(analysis)
}
