calculate_residual_degrees_of_freedom <- function(linear_model) {
 number_of_observations <- nobs(linear_model)
 number_of_variables <- length(names(linear_model$coefficients))
 regression_degrees_of_freedom <- number_of_observations - number_of_variables
 return(regression_degrees_of_freedom)
}
