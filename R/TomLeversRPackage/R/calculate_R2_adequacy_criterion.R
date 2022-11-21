#' @export
calculate_R2_adequacy_criterion <- function(full_linear_model) {
 summary_of_full_linear_model <- summary(full_linear_model)
 r_squared <- summary_of_full_linear_model$r.squared
 significance_level <- 0.05
 regression_degrees_of_freedom <- calculate_regression_degrees_of_freedom(full_linear_model)
 residual_degrees_of_freedom <- calculate_residual_degrees_of_freedom(full_linear_model)
 critical_value_Fc <- calculate_critical_value_Fc(significance_level, regression_degrees_of_freedom, residual_degrees_of_freedom)
 d <- regression_degrees_of_freedom * critical_value_Fc / residual_degrees_of_freedom
 R_squared_adequacy_criterion <- 1 - (1 - r_squared) * (1 + d)
 return(R_squared_adequacy_criterion)
}
