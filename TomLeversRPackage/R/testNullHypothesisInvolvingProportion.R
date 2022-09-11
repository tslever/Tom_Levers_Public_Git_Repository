#' @title testNullHypothesisInvolvingProportion
#' @description Tests a null hypothesis involving a proportion
#' @param population_proportion The population proportion in the null hypothesis
#' @param sample_proportion A sample proportion
#' @param comparator "=", "<=", or ">="
#' @return testResult A message rejecting or not rejecting the null hypothesis, and supporting or not supporting the alternate hypothesis
#' @examples testResult <- testNullHypothesisInvolvingProportion(0.758, 0.505, 467, ">=", 0.05)

#' @export
testNullHypothesisInvolvingProportion <- function(population_proportion, sample_proportion, number_of_data_for_sample, comparator, significance_level) {
    test_statistic_z <- (sample_proportion - population_proportion) / sqrt(population_proportion * (1 - population_proportion) / number_of_data_for_sample)
    if (comparator == "=") {
        probability <- 2 * pnorm(abs(test_statistic_z), 0, 1, lower.tail = FALSE)
    } else if (comparator == ">=") {
        probability <- pnorm(test_statistic_z, 0, 1, lower.tail = TRUE)
    } else if (comparator == "<=") {
        probability <- pnorm(test_statistic_z, 0, 1, lower.tail = FALSE)
    } else {
        stop("You provided an invalid comparator.")
    }
    result <- paste("Since probability ", probability, "\n", sep = "")
    if (probability < significance_level) {
        result <- paste(result, "is less than ", sep = "")
    } else {
        result <- paste(result, "is greater than ", sep = "")
    }
    result <- paste(result, "significance level ", significance_level, ",\nwe ", sep = "")
    if (probability < significance_level) {
        result <- paste(result, "reject ", sep = "")
    } else {
        result <- paste(result, "do not reject ", sep = "")
    }
    result <- paste(result, "the null hypothesis.\nWe ", sep = "")
    if (probability <- significance_level) {
        result <- paste(result, "have ", sep = "")
    } else {
        result <- paste(result, "do not have ", sep = "")
    }
    result <- paste(result, "sufficient evidence to support the alternate hypothesis.", sep = "")
    class(result) <- "HypothesisTestResult"
    return(result)
}

#' @title print.HypothesisTestResult
#' @description Adds to the generic function print the HypothesisTestResult method print.HypothesisTestResult
#' @param the_summary The hypothesis-test-result to print
#' @examples print(the_hypothesis_test_result)
#' @export

print.HypothesisTestResult <- function(the_hypothesis_test_result) {
    cat(the_hypothesis_test_result)
}
