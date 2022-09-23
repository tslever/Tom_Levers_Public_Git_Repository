#' @title analyze_variance
#' @description Analyzes the variance of a linear model
#' @param linear_model The linear model to analyze
#' @return The analysis of variance for the linear model
#' @examples analysis_of_variance <- analyze_variance(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))
#' @import stringr
#' @export

analyze_variance <- function(linear_model) {
    analysis <- capture.output(anova(linear_model))
    regular_expression_for_number <- get_regular_expression_for_number()

    line_with_SSR <- analysis[5]
    DF_SSR_MSR_F0_and_P <- str_extract_all(line_with_SSR, regular_expression_for_number)[[1]]
    DFR <- as.integer(DF_SSR_MSR_F0_and_P[1])
    SSR <- as.double(DF_SSR_MSR_F0_and_P[2])

    line_with_SSRes <- analysis[6]
    DF_SSRes_and_MSRes <- str_extract_all(line_with_SSRes, regular_expression_for_number)[[1]]
    DFRes <- as.integer(DF_SSRes_and_MSRes[1])
    SSRes <- as.double(DF_SSRes_and_MSRes[2])
    DFT <- DFR + DFRes
    SST <- SSR + SSRes
    line_with_SST <- paste("DFT: ", DFT, ", SST: ", SST, sep = "")
    analysis <- append(analysis, line_with_SST)

    coefficient_of_determination_R2 <- SSR / SST
    line_with_R2 <- paste("R2: ", coefficient_of_determination_R2, sep = "")
    analysis <- append(analysis, line_with_R2)

    number_of_observations <- nobs(linear_model)
    line_with_number_of_observations <- paste("Number of observations: ", number_of_observations, sep = "")
    analysis <- append(analysis, line_with_number_of_observations)

    analysis <- paste(analysis, collapse = "\n")
    class(analysis) <- "analysis_of_variance"
    return(analysis)
}


#' @title print.analysis_of_variance
#' @description Adds to the generic function print the analysis_of_variance method print.analysis_of_variance
#' @param analysis The analysis to print
#' @examples print(analysis_of_variance)
#' @export

print.analysis_of_variance <- function(analysis) {
    cat(analysis)
}
