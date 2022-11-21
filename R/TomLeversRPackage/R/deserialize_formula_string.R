#' @export
deserialize_formula_string <- function(formula_string) {
 list_with_vector_with_response_and_serialized_predictors <- strsplit(formula_string, " ~ ")
 vector_with_response_and_serialized_predictors <- list_with_vector_with_response_and_serialized_predictors[[1]]
 response <- vector_with_response_and_serialized_predictors[1]
 serialized_predictors <- vector_with_response_and_serialized_predictors[2]
 list_with_vector_with_predictors <- strsplit(serialized_predictors, " \\+ ")
 vector_of_predictors <- list_with_vector_with_predictors[[1]]
 list_with_response_and_vector_of_predictors <- list(response = response, vector_of_predictors = vector_of_predictors)
 return(reformulate(vector_of_predictors, response))
}
