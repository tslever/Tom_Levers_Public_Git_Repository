#' @title calculateMode
#' @description Calculates the mode given data for a sample
#' @param data_for_sample A vector of data for a sample
#' @return The mode
#' @examples mode <- calculateMode(iris$Sepal.Length)
#' @export

calculateMode <- function(data_for_sample) {
    unique_values <- unique(data_for_sample)
    positions_of_first_matches = match(data_for_sample, unique_values)
    frequencies_of_positions <- tabulate(positions_of_first_matches)
    maximum_frequency <- which.max(frequencies_of_positions)
    return(unique_values[maximum_frequency])
}
