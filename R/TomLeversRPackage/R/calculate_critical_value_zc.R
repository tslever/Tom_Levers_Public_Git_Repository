#' @title calculate_critical_value_zc
#' @description Calculates critical value zc of a standard normal distribution with provided significance level and indicator of whether or not hypothesis test is two tailed
#' @param The significance level
#' @param The indicator of whether of not hypothesis test is two tailed
#' @return The critical value zc
#' @examples critical_value_zc <- calculate_critical_value_zc(0.05, TRUE)

#' @export
calculate_critical_value_zc <- function(significance_level, hypothesis_test_is_two_tailed) {
 if (hypothesis_test_is_two_tailed) {
     critical_value_zc <- qnorm(p = significance_level/2, mean = 0, sd = 1, lower.tail = FALSE)
 } else {
     critical_value_zc <- qnorm(p = significance_level, mean = 0, sd = 1, lower.tail = FALSE)
 }
 return(critical_value_zc)
}
