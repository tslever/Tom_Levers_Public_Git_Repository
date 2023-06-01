#' @export
calculate_AIC <- function(linear_model) {
    number_of_observations <- get_number_of_observations(linear_model)
    residual_sum_of_squares <- calculate_residual_sum_of_squares(linear_model)
    number_of_variables <- get_number_of_variables(linear_model)
    Akaike_Information_Criterion <- number_of_observations * log(residual_sum_of_squares / number_of_observations, base = exp(1)) + 2 * number_of_variables
    return(Akaike_Information_Criterion)
}
