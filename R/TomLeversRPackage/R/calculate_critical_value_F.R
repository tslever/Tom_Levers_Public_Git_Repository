calculate_critical_value_F <- function(linear_model, significance_level) {
 regression_degrees_of_freedom <- calculate_regression_degrees_of_freedom(linear_model)
 residual_degrees_of_freedom <- calculate_residual_degrees_of_freedom(linear_model)
 critical_F_value <- qf(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom, lower.tail = FALSE)
 return(critical_F_value)
}
