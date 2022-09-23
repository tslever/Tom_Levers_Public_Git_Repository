generate_hypothesis_test_result <- function(probability, significance_level) {

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
 class(result) <- "hypothesis_test_result"
 return(result)

}
