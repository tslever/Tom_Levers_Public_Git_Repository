#' @title calculate_critical_value_tc
#' @description Calculates critical value t of a Student's t distribution with a provided number of degrees of freedom with provided significance level and number of confidence intervals
#' @param The significance level
#' @param The number of confidence intervals
#' @param The degrees of freedom of the Student's t distribution
#' @return The critical value t
#' @examples critical_value_t <- calculate_critical_value_tc(0.05, 1, 108)

#' @export
calculate_critical_value_tc <- function(significance_level, number_of_confidence_intervals, residual_degrees_of_freedom) {
 critical_value_t <- qt(significance_level/(2*number_of_confidence_intervals), residual_degrees_of_freedom, lower.tail = FALSE)
 return(critical_value_t)
}
