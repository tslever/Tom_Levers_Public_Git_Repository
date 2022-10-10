#' @title test_null_hypothesis_involving_slope
#' @description Tests a null hypothesis involving a slope
#' @param A significance level
#' @return test_result A message rejecting or not rejecting the null hypothesis, and supporting or not supporting the alternate hypothesis
#' @examples test_result <- test_null_hypothesis_involving_slope(0.05)

#' @export
test_null_hypothesis_involving_multiple_linear_regression_coefficients <- function(linear_model, significance_level) {
    analysis <- anova(linear_model)
    probabilities <- analysis$"Pr(>F)"
    return(generate_hypothesis_test_result(probabilities, significance_level))
}
