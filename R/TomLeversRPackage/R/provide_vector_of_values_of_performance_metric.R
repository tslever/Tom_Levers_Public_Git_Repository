#' @title provide_vector_of_values_of_performance_metric
#' @description Provides vector of values of performance metric
#' @param vector_of_actual_indicators The vector of actual indicators in the set {0, 1}
#' @param vector_of_predicted_probabilities The vector of predicted probabilities in the range [0, 1]
#' @return vector_of_values_of_performance_metric
#' @examples vector_of_values_of_performance_metric <- provide_vector_of_values_of_performance_metric(vector_of_actual_indicators, vector_of_predicted_probabilities)
#' @import dplyr
#' @import rsample

#' @export
provide_vector_of_values_of_performance_metric <- function(vector_of_actual_indicators, vector_of_predicted_probabilities, performance_metric) {
 sequence <- seq(from = 0, to = 1, by = 0.01)
 vector_of_values_of_performance_metric <- rep(0, length(sequence))
 for (i in 1:length(sequence)) {
  vector_of_predicted_indicators <- as.numeric(vector_of_predicted_probabilities > sequence[i])
  if (performance_metric == "accuracy") {
   number_of_true_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 0)
   number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
   number_of_true_predictions <- number_of_true_negatives + number_of_true_positives
   number_of_observations <- length(vector_of_actual_indicators)
   accuracy <- number_of_true_predictions / number_of_observations
   vector_of_values_of_performance_metric[i] <- accuracy
  } else if (performance_metric == "decimal of true positives") {
   number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
   number_of_observations <- length(vector_of_actual_indicators)
   decimal_of_true_positives <- number_of_true_positives / number_of_observations
   vector_of_values_of_performance_metric[i] <- decimal_of_true_positives
  } else if (performance_metric == "error rate") {
   number_of_true_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 0)
   number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
   number_of_true_predictions <- number_of_true_negatives + number_of_true_positives
   number_of_observations <- length(vector_of_actual_indicators)
   accuracy <- number_of_true_predictions / number_of_observations
   error_rate <- 1 - accuracy
   vector_of_values_of_performance_metric[i] <- error_rate
  } else if (performance_metric == "F1 measure") {
   number_of_false_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 0)
   number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
   number_of_positive_predictions <- number_of_true_positives + number_of_false_positives
   precision <- number_of_true_positives / number_of_positive_predictions
   number_of_false_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 1)
   rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
   F1_measure <- 2 / (1/precision + 1/rate_of_true_positives)
   if (is.nan(F1_measure)) {
    F1_measure <- 0
   }
   vector_of_values_of_performance_metric[i] <- F1_measure
  } else if (performance_metric == "FPR") {
   number_of_false_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 0)
   number_of_true_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 0)
   rate_of_false_positives <- number_of_false_positives / (number_of_false_positives + number_of_true_negatives)
   vector_of_values_of_performance_metric[i] <- rate_of_false_positives
  } else if (performance_metric == "number of negatives") {
   number_of_negatives <- sum(vector_of_actual_indicators == 0)
   vector_of_values_of_performance_metric[i] <- number_of_negatives
  } else if (performance_metric == "number of positives") {
   number_of_positives <- sum(vector_of_actual_indicators == 1)
   vector_of_values_of_performance_metric[i] <- number_of_positives
  } else if (performance_metric == "PPV") {
   number_of_false_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 0)
   number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
   number_of_positive_predictions <- number_of_true_positives + number_of_false_positives
   precision <- number_of_true_positives / number_of_positive_predictions
   vector_of_values_of_performance_metric[i] <- precision
  } else if (performance_metric == "threshold") {
   threshold <- sequence[i]
   vector_of_values_of_performance_metric[i] <- threshold
  } else if (performance_metric == "TPR") {
   number_of_true_positives <- sum(vector_of_predicted_indicators == 1 & vector_of_actual_indicators == 1)
   number_of_false_negatives <- sum(vector_of_predicted_indicators == 0 & vector_of_actual_indicators == 1)
   rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
   vector_of_values_of_performance_metric[i] <- rate_of_true_positives
  }
 }
 return(vector_of_values_of_performance_metric)
}
