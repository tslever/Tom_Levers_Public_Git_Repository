#' @title get_column_index
#' @description Gets the index in a data frame given a column name
#' @param The data frame
#' @param The column name
#' @return The column index
#' @examples column_index <- get_column_index(data_frame, "column")

#' @export
get_column_index <- function(data_frame, column_name) {
 column_index <- match(column_name, colnames(data_frame))
 return(column_index)
}
