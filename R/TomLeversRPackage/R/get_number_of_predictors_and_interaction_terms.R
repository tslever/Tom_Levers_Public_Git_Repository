get_number_of_predictors_and_interaction_terms <- function(linear_model) {
 coefficients <- linear_model$coefficients
 names_of_coefficients <- names(coefficients)
 predictors_and_interaction_terms <- names_of_coefficients[2:length(names_of_coefficients)]
 number_of_predictors_and_interaction_terms <- length(predictors_and_interaction_terms)
 return(number_of_predictors_and_interaction_terms)
}
