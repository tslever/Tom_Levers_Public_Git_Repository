calculate_probability <- function(linear_model) {
 F_statistic <- calculate_F_statistic(linear_model)
 regression_degrees_of_freedom <- calculate_regression_degrees_of_freedom(linear_model)
 residual_degrees_of_freedom <- calculate_residual_degrees_of_freedom(linear_model)
 probability <- pf(F_statistic, regression_degrees_of_freedom, residual_degrees_of_freedom, lower.tail = FALSE)
 return(probability)
}
