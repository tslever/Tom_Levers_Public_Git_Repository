#' @title calculate_fraction_of_correct_predictions
#' @description Calculates the fraction of correct predictions for a model given a data set for which to predict
#' @param model The model for which to calculate the fraction of correct predictions given a data set for which to predict
#' @return fraction_of_correct_predictions
#' @examples fraction_of_correct_predictions <- calculate_fraction_of_correct_predictions(model = Direction_vs_Lag2, data_set = Weekly_from_2009_to_2010_inclusive)

#' @export
generate_summary_of_performance <- function(type_of_model, formula, training_data, test_data, K = 1) {
 number_of_test_observations <- nrow(test_data)
 if (type_of_model == "LR") {
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
  vector_of_predicted_directions <- rep("Down", number_of_test_observations)
  condition <- vector_of_predicted_probabilities > 0.5
  vector_of_predicted_directions[condition] = "Up"
  name_of_response <- names(LR_model$model)[1]
 } else if (type_of_model == "LDA") {
  LDA_model <- lda(
   formula = formula,
   data = training_data
  )
  prediction <- predict(LDA_model, newdata = test_data)
  vector_of_predicted_directions <- prediction$class
  name_of_response <- names(attr(LDA_model$terms, "dataClasses"))[1]
 } else if (type_of_model == "QDA") {
  QDA_model <- qda(
   formula = formula,
   data = training_data
  )
  prediction <- predict(QDA_model, newdata = test_data)
  vector_of_predicted_directions <- prediction$class
  name_of_response <- names(attr(QDA_model$terms, "dataClasses"))[1]
 } else if (type_of_model == "KNN") {
  names_of_variables <- all.vars(formula)
  name_of_response <- names_of_variables[1]
  vector_of_names_of_predictors <- names_of_variables[-1]
  matrix_of_values_of_predictors_for_training <- as.matrix(x = training_data[, vector_of_names_of_predictors])
  matrix_of_values_of_predictors_for_testing <- as.matrix(x = test_data[, vector_of_names_of_predictors])
  vector_of_response_values_for_training <- training_data[, name_of_response]
  vector_of_predicted_directions <- knn(
   train = matrix_of_values_of_predictors_for_training,
   test = matrix_of_values_of_predictors_for_testing,
   cl = vector_of_response_values_for_training,
   k = K
  )
 } else {
  stop("A fraction of correct predictions may not be yet calculated for models of the class of parameter model.")
 }
 confusion_matrix <- table(
  vector_of_predicted_directions,
  test_data[, name_of_response]
 )
 map_of_binary_value_to_response_value <- contrasts(x = training_data[, name_of_response])
 number_of_true_negatives <- confusion_matrix[1, 1]
 number_of_false_negatives <- confusion_matrix[1, 2]
 number_of_false_positives <- confusion_matrix[2, 1]
 number_of_true_positives <- confusion_matrix[2, 2]
 number_of_correct_predictions <-
  number_of_true_negatives + number_of_true_positives
 decimal_of_correct_predictions <-
  number_of_correct_predictions / number_of_test_observations
 error_rate <- 1 - decimal_of_correct_predictions
 fraction_of_correct_predictions <- paste(number_of_correct_predictions, " / ", number_of_test_observations, sep = "")
 rate_of_true_positives <- number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
 rate_of_true_negatives <- number_of_true_negatives / (number_of_true_negatives + number_of_false_positives)
 summary_of_performance <- list(
     confusion_matrix = confusion_matrix,
     decimal_of_correct_predictions = decimal_of_correct_predictions,
     error_rate = error_rate,
     fraction_of_correct_predictions = fraction_of_correct_predictions,
     map_of_binary_value_to_response_value = map_of_binary_value_to_response_value,
     equation_for_number_of_true_negatives = paste("TN = CM[1, 1] = ", number_of_true_negatives, sep = ""),
     equation_for_number_of_false_negatives = paste("FN = CM[1, 2] = ", number_of_false_negatives, sep = ""),
     equation_for_number_of_false_positives = paste("FP = CM[2, 1] = ", number_of_false_positives, sep = ""),
     equation_for_number_of_true_positives = paste("TP = CM[2, 2] = ", number_of_true_positives, sep = ""),
     equation_for_true_positive_rate = paste("TPR = Sensitivity = Recall = Hit Rate = TP/P = TP/(TP+FN) = ", rate_of_true_positives, sep = ""),
     equation_for_true_negative_rate = paste("TNR = Specificity = Selectivity = TN/N = TN/(TN+FP) = ", rate_of_true_negatives, sep = "")
 )
 class(summary_of_performance) <- "summary_of_performance"
 return(summary_of_performance)
}
