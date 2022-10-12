calculate_critical_value_t <- function(linear_model, significance_level) {
 regression_degrees_of_freedom <- calculate_regression_degrees_of_freedom(linear_model)
 residual_degrees_of_freedom <- calculate_residual_degrees_of_freedom(linear_model)
 critical_value_t <- qt(significance_level/2, residual_degrees_of_freedom, lower.tail = FALSE)
 return(critical_value_t)
}
