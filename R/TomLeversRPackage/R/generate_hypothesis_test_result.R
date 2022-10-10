generate_hypothesis_test_result <- function(probabilities, significance_level) {

     a_probability_is_greater_than_significance_level <- FALSE
     probability_greater_than_significance_level <- -1.0
     for (probability in probabilities[1:(length(probabilities) - 1)]) { # Works when probabilities is a single number or a vector of numbers
         if (probability > significance_level) {
             a_probability_is_greater_than_significance_level <- TRUE
             probability_greater_than_significance_level <- probability
         }
     }

     result <- ""
     if (a_probability_is_greater_than_significance_level) {
         result <- paste(result, "Since probability ", probability_greater_than_significance_level, " is greater than ", sep = "")
     } else {
         result <- paste(result, "Since no probability is greater than ", sep = "")
     }
     result <- paste(result, "significance level ", significance_level, ",\nwe ", sep = "")
     if (a_probability_is_greater_than_significance_level) {
         result <- paste(result, "do not reject ", sep = "")
     } else {
         result <- paste(result, "reject ", sep = "")
     }
     result <- paste(result, "the null hypothesis.\nWe ", sep = "")
     if (a_probability_is_greater_than_significance_level) {
         result <- paste(result, "do not have ", sep = "")
     } else {
         result <- paste(result, "have ", sep = "")
     }
     result <- paste(result, "sufficient evidence to support the alternate hypothesis.", sep = "")
     class(result) <- "hypothesis_test_result"
     return(result)
}
