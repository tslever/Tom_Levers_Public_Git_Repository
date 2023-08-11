#' @title summarize_performance_of_cross_validated_polynomial_regression_models
#' @description Summarizes performance of cross-validated polynomial regression models
#' @return summary_of_performance The summary of performance of cross-validated polynomial regression models
#' @examples summary_of_performance <- summarize_performance_of_cross_validated_polynomial_regression_models(formula = Direction ~ Lag1 + Lag2, data_frame = ISLR2::Weekly)

#' @export
summarize_performance_of_cross_validated_polynomial_regression_models <- function(formula, data_frame, number_of_folds) {
 number_of_observations <- nrow(data_frame)
 number_of_data_per_fold <- ceiling(number_of_observations / number_of_folds)
 vector_of_mean_squared_errors <- double(0)
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
  mean_squared_error <- summarize_performance_of_one_model(type_of_model, formula, training_data, testing_data)
  vector_of_mean_squared_errors <- append(vector_of_mean_squared_errors, mean_squared_error)
 }
 summary_of_performance <- mean(vector_of_mean_squared_errors)
 return(summary_of_performance)
}
