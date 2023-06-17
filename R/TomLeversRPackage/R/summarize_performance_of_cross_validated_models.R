#' @title summarize_performance_of_cross_validated_models
#' @description Summarizes performance of cross-validated models
#' @param type_of_model The type of model. An element in the set {"Polynomial Regression", Logistic Regression"}.
#' @return summary_of_performance_of_cross_validated_models The summary of performance of cross-validated models
#' @examples summary_of_performance_of_cross_validated_models <- summarize_performance_of_cross_validated_models(type_of_model = "LR", formula = Direction ~ Lag1 + Lag2, data_frame = ISLR2::Weekly)

#' @export
summarize_performance_of_cross_validated_models <- function(type_of_model, formula, data_frame, number_of_folds) {
 number_of_observations <- nrow(data_frame)
 number_of_data_per_fold <- ceiling(number_of_observations / number_of_folds)
 if (type_of_model == "Polynomial Regression") {
  vector_of_mean_squared_errors <- double(0)
 } else if (type_of_model == "Logistic Regression") {
  vector_of_rates_of_errors <- double(0)
  vector_of_rates_of_true_positives <- double(0)
  vector_of_rates_of_false_positives <- double(0)
 }
 for (i in 1:number_of_folds) {
  if (number_of_folds == 1) {
   training_data <- data_frame
   testing_data <- data_frame
  } else {
   if (i < number_of_folds) {
    vector_of_indices_of_observations_in_fold <- c((1+number_of_data_per_fold*(i-1)):(number_of_data_per_fold*i))
    training_data <- data_frame[-vector_of_indices_of_observations_in_fold, ]
    testing_data <- data_frame[vector_of_indices_of_observations_in_fold, ]
   } else {
    #i <- number_of_folds
    vector_of_indices_of_observations_in_fold <- c((1+number_of_data_per_fold*(i-1)):number_of_observations)
    training_data <- data_frame[-vector_of_indices_of_observations_in_fold, ]
    testing_data <- data_frame[vector_of_indices_of_observations_in_fold, ]
   }
  }
  if (type_of_model == "Polynomial Regression") {
   mean_squared_error <- summarize_performance_of_one_model(type_of_model, formula, training_data, testing_data)
   vector_of_mean_squared_errors <- append(vector_of_mean_squared_errors, mean_squared_error)
  } else if (type_of_model == "Logistic Regression") {
   summary_of_performance <- summarize_performance_of_one_model(
    type_of_model = type_of_model,
    formula = formula,
    training_data = training_data,
    test_data = testing_data
   )
   vector_of_rates_of_errors <- append(vector_of_rates_of_errors, summary_of_performance$rate_of_errors)
   vector_of_rates_of_true_positives <- append(vector_of_rates_of_true_positives, summary_of_performance$rate_of_true_positives)
   vector_of_rates_of_false_positives <- append(vector_of_rates_of_false_positives, summary_of_performance$rate_of_false_positives)
  } else {
   error_message <- paste("CV error rate may not be calculated yet for type of model ", type_of_model, sep = "")
   stop(error_message)
  }
 }
 if (type_of_model == "Polynomial Regression") {
  summary_of_performance_after_CV <- mean(vector_of_mean_squared_errors)
 } else if (type_of_model == "Logistic Regression") {
  mean_rate_of_errors <- mean(vector_of_rates_of_errors)
  mean_rate_of_false_positives <- mean(vector_of_rates_of_false_positives)
  mean_rate_of_true_positives <- mean(vector_of_rates_of_true_positives)
  summary_of_performance_of_cross_validated_models <- list(
   mean_rate_of_errors = mean_rate_of_errors,
   mean_rate_of_false_positives = mean_rate_of_false_positives,
   mean_rate_of_true_positives = mean_rate_of_true_positives
  )
 }
 return(summary_of_performance_of_cross_validated_models)
}
