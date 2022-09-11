#' @title analyzeVariance
#' @description Analyzes variance
#' @param linear_model The linear model to analyze
#' @return The analysis of variance for the linear model
#' @examples the_analysis_of_variance <- analyzeVariance(the_linear_model)
#' @import stringr
#' @export

analyzeVariance <- function(linear_model) {
    the_summary <- capture.output(anova(linear_model))
    the_regular_expression_for_a_decimal <- "-?\\d+(\\.\\d+)?"

    the_line_with_SSR <- the_summary[5]
    DF_SSR_MSR_F0_and_P <- str_extract_all(the_line_with_SSR, the_regular_expression_for_a_decimal)[[1]]
    SSR <- as.double(DF_SSR_MSR_F0_and_P[2])
    the_line_with_SSRes <- the_summary[6]
    DF_SSRes_MSRes <- str_extract_all(the_line_with_SSRes, the_regular_expression_for_a_decimal)[[1]]
    SSRes <- as.double(DF_SSRes_MSRes[2])
    SST <- SSR + SSRes
    the_line_with_SST <- paste("SST: ", SST, sep = "")
    the_summary <- append(the_summary, the_line_with_SST)

    the_coefficient_of_determination_R2 <- SSR / SST
    the_line_with_the_coefficient_of_determination_R2 <- paste("R2: ", the_coefficient_of_determination_R2, sep = "")
    the_summary <- append(the_summary, the_line_with_the_coefficient_of_determination_R2)

    the_number_of_observations <- nobs(linear_model)
    the_line_with_the_number_of_observations <- paste("Number of observations: ", the_number_of_observations, sep = "")
    the_summary <- append(the_summary, the_line_with_the_number_of_observations)

    the_summary <- paste(the_summary, collapse = "\n")
    class(the_summary) <- "AnalysisOfVariance"
    return(the_summary)
}


#' @title print.linearModelSummary
#' @description Adds to the generic function print the LinearModelSummary method print.LinearModelSummary
#' @param the_summary The linear-model summary to print
#' @examples print(the_linear_model_summary)
#' @export

print.AnalysisOfVariance <- function(the_analysis_of_variance) {
 cat(the_analysis_of_variance)
}
