#' @title calculate_p_value_from_F_statistic_and_regression_and_residual_degrees_of_freedom
#' @description Calculates the p-value corresponding to a provided test statistic F0 of an F distribution with provided numbers of regression and residual degrees of freedom
#' @param The test statistic F0
#' @param The regression degrees of freedom
#' @param The residual degrees of freedom
#' @return The p-value corresponding to a provided test statistic F0 of an F distribution with provided numbers of regression and residual degrees of freedom
#' @examples p_value <- calculate_p_value_from_F_statistic_and_regression_and_residual_degrees_of_freedom(0.586, 5, 29)

#' @export
calculate_p_value_from_F_statistic_and_regression_and_residual_degrees_of_freedom <- function(F_statistic, regression_degrees_of_freedom, residual_degrees_of_freedom) {
 probability <- pf(F_statistic, regression_degrees_of_freedom, residual_degrees_of_freedom, lower.tail = FALSE)
 return(probability)
}
