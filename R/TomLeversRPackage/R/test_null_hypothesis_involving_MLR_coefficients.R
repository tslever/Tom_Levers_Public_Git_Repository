#' @title test_null_hypothesis_involving_MLR_coefficients
#' @description Tests a null hypothesis involving the coefficients of a Multiple Linear Regression model
#' @param A Multiple Linear Regression model
#' @param A significance level
#' @return test_result A message rejecting or not rejecting the null hypothesis, and supporting or not supporting the alternate hypothesis
#' @examples test_result <- test_null_hypothesis_involving_MLR_coefficients(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris), 0.05)

#' @export
test_null_hypothesis_involving_MLR_coefficients <- function(linear_model, significance_level) {
    probability <- calculate_p_value_for_all_MLR_coefficients(linear_model)
    return(generate_hypothesis_test_result(probability, significance_level))
}
