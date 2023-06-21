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
 vector_of_thresholds <- double(0)
 vector_of_rates_of_true_positives <- double(0)
 vector_of_rates_of_false_positives <- double(0)
 for (i in seq(from = 0, to = 1, by = 0.01)) {
  vector_of_predicted_indicators <- as.numeric(vector_of_predicted_probabilities > i)
  number_of_false_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 1)
  number_of_false_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 0)
  number_of_true_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 0)
  number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
  # TPR = Sensitivity = Recall = Hit Rate = TP/P = TP/(TP+FN)
  rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
  # FPR = Fall-Out = FP/N = FP/(FP+TN)
  rate_of_false_positives <- number_of_false_positives / (number_of_false_positives + number_of_true_negatives)
  vector_of_thresholds <- append(vector_of_thresholds, i)
  vector_of_rates_of_true_positives <- append(vector_of_rates_of_true_positives, rate_of_true_positives)
  vector_of_rates_of_false_positives <- append(vector_of_rates_of_false_positives, rate_of_false_positives)
 }
 data_frame_of_thresholds_TPRs_and_FPRs <- data.frame(
  threshold = vector_of_thresholds,
  TPR = vector_of_rates_of_true_positives,
  FPR = vector_of_rates_of_false_positives
 )
 return(data_frame_of_thresholds_TPRs_and_FPRs)
}
