#' @title construct_confidence_interval_for_difference_between_two_population_means
#' @description Constructs a confidence interval for the difference between two population means based on two sets of sample statistics
#' @param data_for_sample_1 A vector of numerical data for one sample
#' @param data_for_sample_2 A vector of numerical data for the other sample
#' @param significance_level A significance level (e.g., 0.05)
#' @return A confidence interval
#' @examples confidence_interval <- construct_confidence_interval_for_difference_between_two_population_means(iris %>% filter(Species == "versicolor") %>% pull(Sepal.Length), iris %>% filter(Species == "setosa") %>% pull(Sepal.Length), 0.05)

#' @export
construct_confidence_interval_for_difference_between_two_population_means <- function(data_for_sample_1, data_for_sample_2, significance_level) {
    number_of_data_for_sample_1 <- length(data_for_sample_1)
    number_of_data_for_sample_2 <- length(data_for_sample_2)
    degrees_of_freedom_for_sample_1 <- number_of_data_for_sample_1 - 1
    degrees_of_freedom_for_sample_2 <- number_of_data_for_sample_2 - 1
    degrees_of_freedom <- degrees_of_freedom_for_sample_1 + degrees_of_freedom_for_sample_2
    mean_for_sample_1 <- mean(data_for_sample_1)
    mean_for_sample_2 <- mean(data_for_sample_2)
    difference_between_sample_means <- mean_for_sample_1 - mean_for_sample_2
    variance_for_sample_1 <- var(data_for_sample_1)
    variance_for_sample_2 <- var(data_for_sample_2)
    pooled_standard_deviation = sqrt((degrees_of_freedom_for_sample_1 * variance_for_sample_1 + degrees_of_freedom_for_sample_2 * variance_for_sample_2) / (degrees_of_freedom_for_sample_1 + degrees_of_freedom_for_sample_2))
    standard_error_of_difference_between_sample_means <- pooled_standard_deviation * sqrt(1 / number_of_data_for_sample_1 + 1 / number_of_data_for_sample_2)
    critical_value <- qt(p = significance_level / 2, df = degrees_of_freedom, lower.tail = FALSE)
    margin_of_error <- critical_value * standard_error_of_difference_between_sample_means
    lower_bound <- difference_between_sample_means - margin_of_error
    upper_bound <- difference_between_sample_means + margin_of_error
    return(c(lower_bound, upper_bound))
}
