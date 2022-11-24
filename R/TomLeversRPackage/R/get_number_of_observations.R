get_number_of_observations <- function(linear_model) {
    predicted_values <- linear_model$fitted.values
    number_of_observations <- length(predicted_values)
    return(number_of_observations)
}
