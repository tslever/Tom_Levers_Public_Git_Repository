#' @title plot_decimal_of_correct_predictions_vs_K
#' @description Generates a summary of performance for a model given a data set for which to predict
#' @param type_of_model The type of model. An element in the set {"LR", "LDA", "QDA", "KNN"}.
#' @param formula The formula for the model
#' @param training_data A data frame of training data
#' @param testing_data A data frame of test data
#' @param K The parameter of a K Nearest Neighbors model
#' @return summary_of_performance The summary of performance
#' @examples summary_of_performance <- generate_summary_of_performance(type_of_model = "KNN", formula = Direction ~ Lag1 + Lag2, training_data = ISLR2::Weekly_from_1990_to_2008_inclusive, testing_data = ISLR2::Weekly_from_2009_to_2010_inclusive, K = 350)
#' @import class

#' @export
optimize_K <- function(formula, training_data, testing_data) {

 number_of_training_observations <- nrow(training_data)
 number_of_testing_observations <- nrow(testing_data)
 names_of_variables <- all.vars(formula)
 name_of_response <- names_of_variables[1]
 vector_of_names_of_predictors <- names_of_variables[-1]
 matrix_of_values_of_predictors_for_training <- as.matrix(x = training_data[, vector_of_names_of_predictors])
 matrix_of_values_of_predictors_for_testing <- as.matrix(x = testing_data[, vector_of_names_of_predictors])
 vector_of_response_values_for_training <- training_data[, name_of_response]
 vector_of_values_of_K <- integer(0)
 vector_of_decimals_of_correct_predictions <- double(0)
 vector_of_testing_error_rates <- double(0)
 for (K in seq(from = 1, to = number_of_training_observations, by = floor(number_of_training_observations / 100))) {
  vector_of_predicted_values <- knn(
   train = matrix_of_values_of_predictors_for_training,
   test = matrix_of_values_of_predictors_for_testing,
   cl = vector_of_response_values_for_training,
   k = K
  )
  confusion_matrix <- table(vector_of_predicted_values, testing_data[, name_of_response])
  map_of_binary_value_to_response_value <- contrasts(x = training_data[, name_of_response])
  number_of_true_negatives <- confusion_matrix[1, 1]
  #number_of_false_negatives <- confusion_matrix[1, 2]
  #number_of_false_positives <- confusion_matrix[2, 1]
  number_of_true_positives <- confusion_matrix[2, 2]
  number_of_correct_predictions <- number_of_true_negatives + number_of_true_positives
  decimal_of_correct_predictions <- number_of_correct_predictions / number_of_testing_observations
  testing_error_rate <- 1 - decimal_of_correct_predictions
  vector_of_values_of_K <- append(vector_of_values_of_K, K)
  vector_of_decimals_of_correct_predictions <- append(vector_of_decimals_of_correct_predictions, decimal_of_correct_predictions)
  vector_of_testing_error_rates <- append(vector_of_testing_error_rates, testing_error_rate)
 }
 data_frame_of_values_of_K_and_decimals_of_correct_predictions <- data.frame(K = vector_of_values_of_K, decimal_of_correct_predictions = vector_of_decimals_of_correct_predictions)
 maximum_decimal_of_correct_predictions <- max(vector_of_decimals_of_correct_predictions)
 are_the_maximum_decimals_of_correct_predictions <- vector_of_decimals_of_correct_predictions == maximum_decimal_of_correct_predictions
 vector_of_indices_of_maximum_decimals_of_correct_predictions <- which(are_the_maximum_decimals_of_correct_predictions)
 vector_of_optimal_values_of_K <- vector_of_values_of_K[vector_of_indices_of_maximum_decimals_of_correct_predictions]
 Decimal_Of_Correct_Predictions_Vs_K <- ggplot(
  data = data_frame_of_values_of_K_and_decimals_of_correct_predictions,
  mapping = aes(
   x = K,
   y = decimal_of_correct_predictions
  )
 ) +
  geom_point(alpha = 0.5) +
  labs(
   x = "K",
   y = "decimal of correct predictions",
   title = "Decimal Of Correct Predictions Vs. K"
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
 Error_Rates_Vs_K <- ggplot(
  data = data_frame_of_values_of_K_and_decimals_of_correct_predictions,
  mapping = aes(
   x = K,
   y = vector_of_testing_error_rates
  )
 ) +
  geom_point(alpha = 0.5) +
  labs(
   x = "K",
   y = "testing error rate",
   title = "Testing Error Rate Vs. K"
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
 summary_of_optimizing_K <- list(
  Decimal_Of_Correct_Predictions_Vs_K = Decimal_Of_Correct_Predictions_Vs_K,
  Error_Rates_Vs_K = Error_Rates_Vs_K,
  vector_of_optimal_values_of_K = vector_of_optimal_values_of_K
 )
 class(summary_of_optimizing_K) <- "summary_of_optimizing_K"
 return(summary_of_optimizing_K)
}
