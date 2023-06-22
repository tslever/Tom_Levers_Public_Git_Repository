#' @title provide_data_for_ROC_curve
#' @description Provides data for ROC curve
#' @param vector_of_actual_indicators The vector of actual indicators in the set {0, 1}
#' @param vector_of_predicted_probabilities The vector of predicted probabilities in the range [0, 1]
#' @return data_frame_of_thresholds_TPRs_and_FPRs
#' @examples data_frame_of_thresholds_TPRs_and_FPRs <- provide_data_for_ROC_curve(vector_of_actual_indicators, vector_of_predicted_probabilities)
#' @import dplyr
#' @import rsample

#' @export
provide_data_for_ROC_curve <- function(vector_of_actual_indicators, vector_of_predicted_probabilities) {
 sequence <- seq(from = 0, to = 1, by = 0.01)
 data_frame_of_thresholds_TPRs_and_FPRs <- setNames(
  object = data.frame(
   matrix(
    ncol = 3,
    nrow = length(sequence)
   ),
   nm = c("threshold", "TPR", "FPR")
  )
 )
 for (i in sequence) {
  vector_of_predicted_indicators <- as.numeric(vector_of_predicted_probabilities > i)
  number_of_false_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 1)
  number_of_false_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 0)
  number_of_true_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 0)
  number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
  # TPR = Sensitivity = Recall = Hit Rate = TP/P = TP/(TP+FN)
  rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
  # FPR = Fall-Out = FP/N = FP/(FP+TN)
  rate_of_false_positives <- number_of_false_positives / (number_of_false_positives + number_of_true_negatives)
  data_frame_of_thresholds_TPRs_and_FPRs[i, 1] <- i
  data_frame_of_thresholds_TPRs_and_FPRs[i, 2] <- rate_of_true_positives
  data_frame_of_thresholds_TPRs_and_FPRs[i, 3] <- rate_of_false_positives
 }
 return(data_frame_of_thresholds_TPRs_and_FPRs)
}
