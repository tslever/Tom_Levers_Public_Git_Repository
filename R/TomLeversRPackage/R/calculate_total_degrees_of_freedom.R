calculate_total_degrees_of_freedom <- function(linear_model) {
    number_of_observations <- nobs(linear_model)
    total_degrees_of_freedom <- number_of_observations - 1
    return(total_degrees_of_freedom)
}
