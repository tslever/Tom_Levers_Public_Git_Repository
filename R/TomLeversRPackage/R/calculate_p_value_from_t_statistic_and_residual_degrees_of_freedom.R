#' @title calculate_p_value_from_t_statistic_and_residual_degrees_of_freedom
#' @description Calculates the p-value corresponding to a provided test statistic t of a Student's t distribution with a provided number of degrees of freedom
#' @param The test statistic t
#' @param The residual degrees of freedom
#' @return The p-value corresponding to a provided test statistic t of a Student's t distribution with a provided number of degrees of freedom
#' @examples p_value <- calculate_p_value_from_t_statistic_and_residual_degrees_of_freedom(1.982, 108)

#' @export
calculate_p_value_from_t_statistic_and_residual_degrees_of_freedom <- function(t_statistic, residual_degrees_of_freedom) {
 probability <- pt(abs(t_statistic), residual_degrees_of_freedom, lower.tail = FALSE) * 2
 return(probability)
}
