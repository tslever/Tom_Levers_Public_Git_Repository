#' @title analyze_correlation_matrix
#' @description Analyzes a correlation matrix
#' @param correlation_matrix The correlation matrix to analyze
#' @return The analysis for the correlation matrix
#' @examples analysis <- analyze_correlation_matrix(correlation_matrix)

#' @export
analyze_correlation_matrix <- function(correlation_matrix) {
    predictors <- colnames(correlation_matrix)
    for (i in 1:length(predictors)) {
     predictors_with_very_high_correlations <- character()
     predictors_with_high_correlations <- character()
     predictors_with_moderate_correlations <- character()
     predictors_with_low_correlations <- character()
     predictors_with_negligible_correlations <- character()
     for (j in 1:length(predictors)) {
      if (abs(correlation_matrix[i, j]) > 0.9) {
       predictors_with_very_high_correlations <- append(predictors_with_very_high_correlations, predictors[j])
      } else if (abs(correlation_matrix[i, j]) > 0.7) {
       predictors_with_high_correlations <- append(predictors_with_high_correlations, predictors[j])
      } else if (abs(correlation_matrix[i, j]) > 0.5) {
       predictors_with_moderate_correlations <- append(predictors_with_moderate_correlations, predictors[j])
      } else if (abs(correlation_matrix[i, j]) > 0.3) {
       predictors_with_low_correlations <- append(predictors_with_low_correlations, predictors[j])
      } else {
       predictors_with_negligible_correlations <- append(predictors_with_negligible_correlations, predictors[j])
      }
     }
     predictors_with_very_high_correlations <- paste(predictors_with_very_high_correlations, collapse = ", ")
     predictors_with_high_correlations <- paste(predictors_with_high_correlations, collapse = ", ")
     predictors_with_moderate_correlations <- paste(predictors_with_moderate_correlations, collapse = ", ")
     predictors_with_low_correlations <- paste(predictors_with_low_correlations, collapse = ", ")
     predictors_with_negligible_correlations <- paste(predictors_with_negligible_correlations, collapse = ", ")
     line_with_predictors_with_very_high_correlations <- paste("V: ", predictors_with_very_high_correlations, "\n")
     line_with_predictors_with_high_correlations <- paste("H: ", predictors_with_high_correlations, "\n")
     line_with_predictors_with_moderate_correlations <- paste("M: ", predictors_with_moderate_correlations, "\n")
     line_with_predictors_with_low_correlations <- paste("L: ", predictors_with_low_correlations, "\n")
     line_with_predictors_with_negligible_correlations <- paste("N: ", predictors_with_negligible_correlations, "\n")
     cat(
      paste(
       predictors[i], "\n",
       "    ", line_with_predictors_with_very_high_correlations,
       "    ", line_with_predictors_with_high_correlations,
       "    ", line_with_predictors_with_moderate_correlations,
       "    ", line_with_predictors_with_low_correlations,
       "    ", line_with_predictors_with_negligible_correlations,
       sep = ""
      )
     )
    }
}
