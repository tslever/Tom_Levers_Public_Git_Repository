#' @title summarize_performance
#' @description Generates a summary of performance for a model given a data set for which to predict
#' @param type_of_model The type of model. An element in the set {"Polynomial Regression", "Logistic Regression", "LDA", "QDA", "KNN"}.
#' @param formula The formula for the model
#' @param training_data A data frame of training data
#' @param test_data A data frame of test data
#' @param K The parameter of a K Nearest Neighbors model
#' @return summary_of_performance The summary of performance
#' @examples summary_of_performance <- generate_summary_of_performance(type_of_model = "KNN", formula = Direction ~ Lag1 + Lag2, training_data = ISLR2::Weekly_from_1990_to_2008_inclusive, test_data = ISLR2::Weekly_from_2009_to_2010_inclusive, K = 350)
#' @import class

#' @export
summarize_performance <- function(type_of_model, formula, training_data, test_data, K = 1) {
 number_of_test_observations <- nrow(test_data)
 if (type_of_model == "Polynomial Regression") {
  linear_regression_model <- glm(
   formula = formula,
   data = training_data
  )
  mean_squared_error <- calculate_mean_squared_error(linear_regression_model)
  return(mean_squared_error)
 } else if (type_of_model == "Logistic Regression") {
  LR_model <- glm(
   formula = formula,
   data = training_data,
   family = binomial
  )
  vector_of_predicted_probabilities <- predict(
   object = LR_model,
   newdata = test_data,
   type = "response"
  )
  name_of_response <- names(LR_model$model)[1]
  factor_of_response_values <- test_data[, name_of_response]
  vector_of_levels <- attr(factor_of_response_values, "levels")
  lower_level <- vector_of_levels[1]
  upper_level <- vector_of_levels[2]
  vector_of_predicted_response_values <- rep(lower_level, number_of_test_observations)
  condition <- vector_of_predicted_probabilities > 0.5
  vector_of_predicted_response_values[condition] <- upper_level
 } else if (type_of_model == "LDA") {
  LDA_model <- lda(
   formula = formula,
   data = training_data
  )
  prediction <- predict(LDA_model, newdata = test_data)
  vector_of_predicted_response_values <- prediction$class
  name_of_response <- names(attr(LDA_model$terms, "dataClasses"))[1]
 } else if (type_of_model == "QDA") {
  QDA_model <- qda(
   formula = formula,
   data = training_data
  )
  prediction <- predict(QDA_model, newdata = test_data)
  vector_of_predicted_response_values <- prediction$class
  name_of_response <- names(attr(QDA_model$terms, "dataClasses"))[1]
 } else if (type_of_model == "KNN") {
  names_of_variables <- all.vars(formula)
  name_of_response <- names_of_variables[1]
  vector_of_names_of_predictors <- names_of_variables[-1]
  matrix_of_values_of_predictors_for_training <- as.matrix(x = training_data[, vector_of_names_of_predictors])
  matrix_of_values_of_predictors_for_testing <- as.matrix(x = test_data[, vector_of_names_of_predictors])
  vector_of_response_values_for_training <- training_data[, name_of_response]
  vector_of_predicted_response_values <- knn(
   train = matrix_of_values_of_predictors_for_training,
   test = matrix_of_values_of_predictors_for_testing,
   cl = vector_of_response_values_for_training,
   k = K
  )
 } else {
  error_message <- paste("A fraction of correct predictions may not be yet calculated for models of type ", type_of_model, sep = "")
  stop(error_message)
 }
 confusion_matrix <- table(
  vector_of_predicted_response_values,
  test_data[, name_of_response]
 )
 map_of_binary_value_to_response_value <- contrasts(x = training_data[, name_of_response])
 number_of_false_negatives <- 0
 number_of_false_positives <- 0
 number_of_true_negatives <- 0
 number_of_true_positives <- 0
 for (j in 1:number_of_test_observations) {
  if (vector_of_predicted_response_values[j] == 0 & test_data[j, name_of_response] == 1) {
      number_of_false_negatives <- number_of_false_negatives + 1
  } else if (vector_of_predicted_response_values[j] == 1 & test_data[j, name_of_response] == 0) {
      number_of_false_positives <- number_of_false_positives + 1
  } else if (vector_of_predicted_response_values[j] == 0 & test_data[j, name_of_response] == 0) {
      number_of_true_negatives <- number_of_true_negatives + 1
  } else if (vector_of_predicted_response_values[j] == 1 & test_data[j, name_of_response] == 1) {
      number_of_true_positives <- number_of_true_positives + 1
  }
 }
 number_of_correct_predictions <-
  number_of_true_negatives + number_of_true_positives
 decimal_of_correct_predictions <-
  number_of_correct_predictions / number_of_test_observations
 rate_of_errors <- 1 - decimal_of_correct_predictions
 fraction_of_correct_predictions <- paste(number_of_correct_predictions, " / ", number_of_test_observations, sep = "")
 rate_of_false_positives <- number_of_false_positives / (number_of_false_positives + number_of_true_negatives)
 rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
 rate_of_true_negatives <- number_of_true_negatives / (number_of_true_negatives + number_of_false_positives)
 summary_of_performance <- list(
  confusion_matrix = confusion_matrix,
  decimal_of_correct_predictions = decimal_of_correct_predictions,
  rate_of_errors = rate_of_errors,
  fraction_of_correct_predictions = fraction_of_correct_predictions,
  map_of_binary_value_to_response_value = map_of_binary_value_to_response_value,
  equation_for_number_of_true_negatives = paste("TN = CM[1, 1] = ", number_of_true_negatives, sep = ""),
  equation_for_number_of_false_negatives = paste("FN = CM[1, 2] = ", number_of_false_negatives, sep = ""),
  equation_for_number_of_false_positives = paste("FP = CM[2, 1] = ", number_of_false_positives, sep = ""),
  equation_for_number_of_true_positives = paste("TP = CM[2, 2] = ", number_of_true_positives, sep = ""),
  equation_for_false_positive_rate = paste("FPR = Fall-Out = FP/N = FP/(FP+TN) = ", sep = ""),
  rate_of_false_positives = rate_of_false_positives,
  equation_for_true_positive_rate = paste("TPR = Sensitivity = Recall = Hit Rate = TP/P = TP/(TP+FN) = ", sep = ""),
  rate_of_true_positives = rate_of_true_positives,
  equation_for_true_negative_rate = paste("TNR = Specificity = Selectivity = TN/N = TN/(TN+FP) = ", rate_of_true_negatives, sep = "")
 )
 class(summary_of_performance) <- "summary_of_performance"
 return(summary_of_performance)
}
