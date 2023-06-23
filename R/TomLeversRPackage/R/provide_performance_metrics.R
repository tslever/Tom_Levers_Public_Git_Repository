#' @title provide_performance_metrics
#' @description Provides performance metrics
#' @param vector_of_actual_indicators The vector of actual indicators in the set {0, 1}
#' @param vector_of_predicted_probabilities The vector of predicted probabilities in the range [0, 1]
#' @return data_frame_of_performance_metrics
#' @examples data_frame_of_performance_metrics <- provide_performance_metrics(vector_of_actual_indicators, vector_of_predicted_probabilities)
#' @import dplyr
#' @import rsample

#' @export
provide_performance_metrics <- function(vector_of_actual_indicators, vector_of_predicted_probabilities) {
 sequence <- seq(from = 0, to = 1, by = 0.01)
 data_frame_of_performance_metrics <- setNames(
  object = data.frame(
   matrix(
    ncol = 5,
    nrow = length(sequence)
   )
  ),
  nm = c("threshold", "TPR", "FPR", "PPV", "decimal_of_true_positives")
 )
 number_of_observations <- length(vector_of_actual_indicators)
 for (i in 1:length(sequence)) {
  vector_of_predicted_indicators <- as.numeric(vector_of_predicted_probabilities > sequence[i])
  number_of_false_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 1)
  number_of_false_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 0)
  number_of_true_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 0)
  number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
  # TPR = Sensitivity = Recall = Hit Rate = TP/P = TP/(TP+FN)
  rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
  # FPR = Fall-Out = FP/N = FP/(FP+TN)
  rate_of_false_positives <- number_of_false_positives / (number_of_false_positives + number_of_true_negatives)
  # PPV = Precision = Positive Predictive Value = TP/\hat{P} = TP/(TP+FP)
  number_of_positive_predictions <- number_of_true_positives + number_of_false_positives
  precision <- number_of_true_positives / number_of_positive_predictions
  #decimal_of_predicted_positives <- number_of_positive_predictions / number_of_observations
  decimal_of_true_positives <- number_of_true_positives / number_of_observations
  data_frame_of_performance_metrics[i, 1] <- sequence[i]
  data_frame_of_performance_metrics[i, 2] <- rate_of_true_positives
  data_frame_of_performance_metrics[i, 3] <- rate_of_false_positives
  data_frame_of_performance_metrics[i, 4] <- precision
  data_frame_of_performance_metrics[i, 5] <- decimal_of_true_positives
 }
 return(data_frame_of_performance_metrics)
}
