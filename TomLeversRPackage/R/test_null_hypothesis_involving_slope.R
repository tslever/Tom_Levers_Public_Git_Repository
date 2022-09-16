#' @title test_null_hypothesis_involving_slope
#' @description Tests a null hypothesis involving a slope
#' @param A significance level
#' @return test_result A message rejecting or not rejecting the null hypothesis, and supporting or not supporting the alternate hypothesis
#' @examples test_result <- test_null_hypothesis_involving_slope(0.05)

#' @export
test_null_hypothesis_involving_slope <- function(linear_model, significance_level) {
    analysis <- capture.output(anova(linear_model))
    regular_expression_for_number <- get_regular_expression_for_number()
    line_with_SSR <- analysis[5]
    DF_SSR_MSR_F0_and_P <- str_extract_all(line_with_SSR, regular_expression_for_number)[[1]]
    probability <- as.double(DF_SSR_MSR_F0_and_P[5])
    return(generate_hypothesis_test_result(probability, significance_level))
}
