#' @title get_index_of_label_of_vector
#' @description Gets the index of a label in the vector of names of a vector
#' @param The vector
#' @param The label
#' @return The index of the label
#' @examples index_of_label <- get_index_of_label_of_vector(vector, "label")

#' @export
get_index_of_label_of_vector <- function(vector, label) {
 index_of_label <- match(label, names(vector))
 return(index_of_label)
}
