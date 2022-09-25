#' @title construct_confidence_interval_for_population_mean
#' @description Constructs a confidence interval for a population mean based on a set of statistics for a sample
#' @param data_for_sample A vector of data for a sample
#' @param significance_level A significance level (e.g., 0.05)
#' @return A confidence interval
#' @examples confidence_interval <- construct_confidence_interval_for_population_mean(iris$Sepal.Length, 0.05)

#' @export
construct_confidence_interval_for_population_mean <- function(data_for_sample, significance_level) {
    number_of_data <- length(data_for_sample)
    degrees_of_freedom <- number_of_data - 1
    mean_value <- mean(data_for_sample)
    standard_deviation <- sd(data_for_sample)
    standard_error_of_sample_mean <- standard_deviation / sqrt(number_of_data)
    critical_value <- qt(p = significance_level / 2, df = degrees_of_freedom, lower.tail = FALSE)
    margin_of_error <- critical_value * standard_error_of_sample_mean
    lower_bound <- mean_value - margin_of_error
    upper_bound <- mean_value + margin_of_error
    return(c(lower_bound, upper_bound))
}
