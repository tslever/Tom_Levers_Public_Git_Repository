#' @title analyze_correlation_matrix
#' @description Analyzes a correlation matrix
#' @param correlation_matrix The correlation matrix to analyze
#' @return The analysis for the correlation matrix
#' @examples analysis <- analyze_correlation_matrix(correlation_matrix)

#' @export
analyze_correlation_matrix <- function(correlation_matrix) {
    predictors <- colnames(correlation_matrix)
    for (i in 1:length(predictors)) {
     predictors_with_very_high_positive_correlations <- character()
     predictors_with_very_high_negative_correlations <- character()
     predictors_with_high_positive_correlations <- character()
     predictors_with_high_negative_correlations <- character()
     predictors_with_moderate_positive_correlations <- character()
     predictors_with_moderate_negative_correlations <- character()
     predictors_with_low_positive_correlations <- character()
     predictors_with_low_negative_correlations <- character()
     predictors_with_negligible_correlations <- character()
     for (j in 1:length(predictors)) {
      if (correlation_matrix[i, j] > 0.9) {
       predictors_with_very_high_positive_correlations <- append(predictors_with_very_high_positive_correlations, predictors[j])
      } else if (correlation_matrix[i, j] < -0.9) {
       predictors_with_very_high_negative_correlations <- append(predictors_with_very_high_negative_correlations, predictors[j])
      } else if (correlation_matrix[i, j] > 0.7) {
       predictors_with_high_positive_correlations <- append(predictors_with_high_positive_correlations, predictors[j])
      } else if (correlation_matrix[i, j] < -0.7) {
       predictors_with_high_negative_correlations <- append(predictors_with_high_negative_correlations, predictors[j])
      } else if (correlation_matrix[i, j] > 0.5) {
       predictors_with_moderate_positive_correlations <- append(predictors_with_moderate_positive_correlations, predictors[j])
      } else if (correlation_matrix[i, j] < -0.5) {
       predictors_with_moderate_negative_correlations <- append(predictors_with_moderate_negative_correlations, predictors[j])
      } else if (correlation_matrix[i, j] > 0.3) {
       predictors_with_low_positive_correlations <- append(predictors_with_low_positive_correlations, predictors[j])
      } else if (correlation_matrix[i, j] < -0.3) {
       predictors_with_low_negative_correlations <- append(predictors_with_low_negative_correlations, predictors[j])
      } else {
       predictors_with_negligible_correlations <- append(predictors_with_negligible_correlations, predictors[j])
      }
     }
     predictors_with_very_high_positive_correlations <- paste(predictors_with_very_high_positive_correlations, collapse = ", ")
     predictors_with_very_high_negative_correlations <- paste(predictors_with_very_high_negative_correlations, collapse = ", ")
     predictors_with_high_positive_correlations <- paste(predictors_with_high_positive_correlations, collapse = ", ")
     predictors_with_high_negative_correlations <- paste(predictors_with_high_negative_correlations, collapse = ", ")
     predictors_with_moderate_positive_correlations <- paste(predictors_with_moderate_positive_correlations, collapse = ", ")
     predictors_with_moderate_negative_correlations <- paste(predictors_with_moderate_negative_correlations, collapse = ", ")
     predictors_with_low_positive_correlations <- paste(predictors_with_low_positive_correlations, collapse = ", ")
     predictors_with_low_negative_correlations <- paste(predictors_with_low_negative_correlations, collapse = ", ")
     predictors_with_negligible_correlations <- paste(predictors_with_negligible_correlations, collapse = ", ")
     line_with_predictors_with_very_high_positive_correlations <- paste("V+: ", predictors_with_very_high_positive_correlations, "\n")
     line_with_predictors_with_very_high_negative_correlations <- paste("V-: ", predictors_with_very_high_negative_correlations, "\n")
     line_with_predictors_with_high_positive_correlations <- paste("H+: ", predictors_with_high_positive_correlations, "\n")
     line_with_predictors_with_high_negative_correlations <- paste("H-: ", predictors_with_high_negative_correlations, "\n")
     line_with_predictors_with_moderate_positive_correlations <- paste("M+: ", predictors_with_moderate_positive_correlations, "\n")
     line_with_predictors_with_moderate_negative_correlations <- paste("M-: ", predictors_with_moderate_negative_correlations, "\n")
     line_with_predictors_with_low_positive_correlations <- paste("L+: ", predictors_with_low_positive_correlations, "\n")
     line_with_predictors_with_low_negative_correlations <- paste("L-: ", predictors_with_low_negative_correlations, "\n")
     line_with_predictors_with_negligible_correlations <- paste("N: ", predictors_with_negligible_correlations, "\n")
     cat(
      paste(
       predictors[i], "\n",
       "    ", line_with_predictors_with_very_high_positive_correlations,
       "    ", line_with_predictors_with_very_high_negative_correlations,
       "    ", line_with_predictors_with_high_positive_correlations,
       "    ", line_with_predictors_with_high_negative_correlations,
       "    ", line_with_predictors_with_moderate_positive_correlations,
       "    ", line_with_predictors_with_moderate_negative_correlations,
       "    ", line_with_predictors_with_low_positive_correlations,
       "    ", line_with_predictors_with_low_negative_correlations,
       "    ", line_with_predictors_with_negligible_correlations,
       sep = ""
      )
     )
    }
}
