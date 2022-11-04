#' @title convert_to_categorical_vector
#' @description Converts a discrete quantitative vector to a categorical vector
#' @param A discrete quantitative vector
#' @return A categorical vector
#' @examples categorical_vector <- convert_to_categorical_vector(birthwt$race) # MASS::birthwt

#' @export
convert_to_categorical_vector <- function(discrete_quantitative_vector, vector_of_levels) {
 categorical_vector <- factor(discrete_quantitative_vector)
 levels(categorical_vector) <- vector_of_levels
 return(categorical_vector)
}
