#' @title get_row_index
#' @description Gets the index in a data frame given a row name
#' @param The data frame
#' @param The row name
#' @return The row index
#' @examples row_index <- get_row_index(data_frame, "row")

#' @export
get_row_index <- function(data_frame, row_name) {
 row_index <- match(row_name, rownames(data_frame))
 return(row_index)
}
