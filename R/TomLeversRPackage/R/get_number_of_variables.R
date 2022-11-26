#' @export
get_number_of_variables <- function(linear_model) {
    number_of_predictors_and_interaction_terms <- get_number_of_predictors_and_interaction_terms(linear_model)
    number_of_variables <- number_of_predictors_and_interaction_terms + 1
    return(number_of_variables)
}
