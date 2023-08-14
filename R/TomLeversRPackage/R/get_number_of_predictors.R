#' @export
get_number_of_predictors <- function(formula) {
 names_of_variables <- all.vars(formula)
 vector_of_names_of_predictors <- names_of_variables[-1]
 number_of_predictors <- length(vector_of_names_of_predictors)
 return(number_of_predictors)
}
