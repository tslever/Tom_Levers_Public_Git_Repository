#' @title get_index_of_column_of_data_frame
#' @description Gets the index in a data frame given a column name
#' @param The data frame
#' @param The column name
#' @return The column index
#' @examples column_index <- get_index_of_column_of_data_frame(data_frame, "column")

#' @export
get_index_of_column_of_data_frame <- function(data_frame, column_name) {
 column_index <- match(column_name, colnames(data_frame))
 return(column_index)
}
