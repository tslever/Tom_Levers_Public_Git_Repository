#' @export
normalize_vector <- function(vector_to_normalize) {
 normalized_vector <- (vector_to_normalize - min(vector_to_normalize)) / (max(vector_to_normalize) - min(vector_to_normalize))
 return(normalized_vector)
}
