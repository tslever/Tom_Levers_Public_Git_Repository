#' @title calculate_critical_value_Fc
#' @description Calculates critical value Fc of an F distribution with specified significance level, number of regression degrees of freedom, and number of residual degrees of freedom
#' @param The significance level
#' @param The regression degrees of freedom
#' @param The residual degrees of freedom
#' @return The critical value F
#' @examples critical_value_F <- calculate_critical_value_Fc(0.05, 2, 52)

#' @export
calculate_critical_value_Fc <- function(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom) {
    critical_F_value <- qf(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom, lower.tail = FALSE)
    return(critical_F_value)
}
