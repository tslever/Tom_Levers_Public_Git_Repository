#' @title plot_decimal_of_correct_predictions_vs_K
#' @description Generates a summary of performance for a model given a data set for which to predict
#' @param type_of_model The type of model. An element in the set {"LR", "LDA", "QDA", "KNN"}.
#' @param formula The formula for the model
#' @param training_data A data frame of training data
#' @param testing_data A data frame of test data
#' @param K The parameter of a K Nearest Neighbors model
#' @return summary_of_performance The summary of performance
#' @examples summary_of_performance <- generate_summary_of_performance(type_of_model = "KNN", formula = Direction ~ Lag1 + Lag2, training_data = ISLR2::Weekly_from_1990_to_2008_inclusive, testing_data = ISLR2::Weekly_from_2009_to_2010_inclusive, K = 350)

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
 vector_of_decimals_of_correct_predictions_for_training_data <- double(0)
 vector_of_decimals_of_correct_predictions_for_testing_data <- double(0)
 vector_of_training_error_rates <- double(0)
 vector_of_testing_error_rates <- double(0)
 increment = floor(number_of_training_observations / 100)
 if (increment %% 2 == 1) {
  increment <- increment - 1
 }
 #for (K in seq(from = 1, to = 100, by = 10)) {
 for (K in seq(from = 1, to = number_of_training_observations / 2, by = increment)) {
  vector_of_predicted_values_for_training_data <- knn(
   train = matrix_of_values_of_predictors_for_training,
   test = matrix_of_values_of_predictors_for_training,
   cl = vector_of_response_values_for_training,
   k = K
  )
  confusion_matrix_for_training_data <- table(vector_of_predicted_values_for_training_data, training_data[, name_of_response])
  number_of_true_negatives_for_training_data <- confusion_matrix_for_training_data[1, 1]
  number_of_true_positives_for_training_data <- confusion_matrix_for_training_data[2, 2]
  number_of_correct_predictions_for_training_data <- number_of_true_negatives_for_training_data + number_of_true_positives_for_training_data
  decimal_of_correct_predictions_for_training_data <- number_of_correct_predictions_for_training_data / number_of_training_observations
  training_error_rate <- 1 - decimal_of_correct_predictions_for_training_data
  vector_of_values_of_K <- append(vector_of_values_of_K, K)
  vector_of_decimals_of_correct_predictions_for_training_data <- append(vector_of_decimals_of_correct_predictions_for_training_data, decimal_of_correct_predictions_for_training_data)
  vector_of_training_error_rates <- append(vector_of_training_error_rates, training_error_rate)
  vector_of_predicted_values_for_testing_data <- knn(
   train = matrix_of_values_of_predictors_for_training,
   test = matrix_of_values_of_predictors_for_testing,
   cl = vector_of_response_values_for_training,
   k = K
  )
  confusion_matrix_for_testing_data <- table(vector_of_predicted_values_for_testing_data, testing_data[, name_of_response])
  number_of_true_negatives_for_testing_data <- confusion_matrix_for_testing_data[1, 1]
  number_of_true_positives_for_testing_data <- confusion_matrix_for_testing_data[2, 2]
  number_of_correct_predictions_for_testing_data <- number_of_true_negatives_for_testing_data + number_of_true_positives_for_testing_data
  decimal_of_correct_predictions_for_testing_data <- number_of_correct_predictions_for_testing_data / number_of_testing_observations
  testing_error_rate <- 1 - decimal_of_correct_predictions_for_testing_data
  vector_of_decimals_of_correct_predictions_for_testing_data <- append(vector_of_decimals_of_correct_predictions_for_testing_data, decimal_of_correct_predictions_for_testing_data)
  vector_of_testing_error_rates <- append(vector_of_testing_error_rates, testing_error_rate)
 }
 data_frame_of_values_of_K_and_decimals_of_correct_predictions <- data.frame(K = vector_of_values_of_K, decimal_of_correct_predictions_for_training_data = vector_of_decimals_of_correct_predictions_for_training_data, decimal_of_correct_predictions_for_testing_data = vector_of_decimals_of_correct_predictions_for_testing_data)
 minimum_testing_error_rate <- min(vector_of_testing_error_rates)
 maximum_decimal_of_correct_predictions_for_testing_data <- max(vector_of_decimals_of_correct_predictions_for_testing_data)
 are_the_maximum_decimals_of_correct_predictions_for_testing_data <- vector_of_decimals_of_correct_predictions_for_testing_data == maximum_decimal_of_correct_predictions_for_testing_data
 vector_of_indices_of_maximum_decimals_of_correct_predictions_for_testing_data <- which(are_the_maximum_decimals_of_correct_predictions_for_testing_data)
 vector_of_optimal_values_of_K_for_testing_data <- vector_of_values_of_K[vector_of_indices_of_maximum_decimals_of_correct_predictions_for_testing_data]
 indices <- integer(0)
 training_decimal <- -1.0
 for (i in vector_of_indices_of_maximum_decimals_of_correct_predictions_for_testing_data) {
  corresponding_decimal_of_correct_predictions_for_training_data <- vector_of_decimals_of_correct_predictions_for_training_data[i]
  #if (corresponding_decimal_of_correct_predictions_for_training_data == training_decimal) {
   #stop("optimize_K yet cannot optimize K when two KNN models have the maximum decimal of correct predictions for testing data and the same decimals of correct predictions for training data.")
  #}
  if (corresponding_decimal_of_correct_predictions_for_training_data >= training_decimal) {
   training_decimal <- corresponding_decimal_of_correct_predictions_for_training_data
   indices <- append(indices, i)
  }
 }
 optimal_K <- vector_of_values_of_K[indices]
 Decimals_Of_Correct_Predictions_Vs_K <- ggplot(
  data = data_frame_of_values_of_K_and_decimals_of_correct_predictions,
  mapping = aes(x = K)
 ) +
  geom_point(aes(y = vector_of_decimals_of_correct_predictions_for_training_data, color = "Training"), alpha = 0.5) +
  geom_point(aes(y = vector_of_decimals_of_correct_predictions_for_testing_data, color = "Testing"), alpha = 0.5) +
  labs(
   x = "K",
   y = "decimals of correct predictions",
   title = "Decimals Of Correct Predictions Vs. K"
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
 Error_Rates_Vs_K <- ggplot(
  data = data_frame_of_values_of_K_and_decimals_of_correct_predictions,
  mapping = aes(x = K)
 ) +
  geom_point(aes(y = vector_of_training_error_rates, color = "Training"), alpha = 0.5) +
  geom_point(aes(y = vector_of_testing_error_rates, color = "Testing"), alpha = 0.5) +
  labs(
   x = "K",
   y = "error rates",
   title = "Error Rates Vs. K"
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
 summary_of_optimizing_K <- list(
  Decimals_Of_Correct_Predictions_Vs_K = Decimals_Of_Correct_Predictions_Vs_K,
  Error_Rates_Vs_K = Error_Rates_Vs_K,
  minimum_testing_error_rate = minimum_testing_error_rate,
  maximum_decimal_of_correct_predictions_for_testing_data = maximum_decimal_of_correct_predictions_for_testing_data,
  optimal_K = optimal_K
 )
 class(summary_of_optimizing_K) <- "summary_of_optimizing_K"
 return(summary_of_optimizing_K)
}
