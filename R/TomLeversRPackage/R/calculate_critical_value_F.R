#' @title calculate_critical_value_F
#' @description Calculates critical value F of an F distribution with provided numbers of regression and residual degrees of freedom with provided significance level
#' @param The significance level
#' @param The regression degrees of freedom
#' @param The residual degrees of freedom
#' @return The critical value F
#' @examples critical_value_F <- calculate_critical_value_F(0.05, 2, 52)

#' @export
calculate_critical_value_F <- function(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom) {
 critical_F_value <- qf(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom, lower.tail = FALSE)
 return(critical_F_value)
}
