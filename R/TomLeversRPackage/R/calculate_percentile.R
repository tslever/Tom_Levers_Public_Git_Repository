#' @title calculate_percentile
#' @description Calculates the percentile of a value given data for a sample
#' @param data_for_sample A vector of data for a sample
#' @param value A number for which to find the percentile given data for a sample
#' @return A percentile
#' @examples percentile <- calculate_percentile(iris$Sepal.Length, 5.8)

#' @export
calculate_percentile <- function(data_for_sample, value) {
    sorted_data_for_sample <- sort(data_for_sample, decreasing = FALSE, na.last = NA, partial = NULL, method = "auto", index.return = FALSE)
    number_of_values_less_than_value <- sum(data_for_sample < value)
    number_of_repetitions <- sum(data_for_sample == value)
    percentile <- round((number_of_values_less_than_value + number_of_repetitions / 2) / length(data_for_sample) * 100, digits = 0)
    return(percentile)
}
