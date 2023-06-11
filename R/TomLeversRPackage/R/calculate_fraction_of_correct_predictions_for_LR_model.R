#' @title calculate_fraction_of_correct_predictions_for_LR_model
#' @description Calculates the fraction of correct predictions for an Logistic Regression model given a data set for which to predict
#' @param LR_model The Logistic Regression model for which to calculate the fraction of correct predictions given a data set for which to predict
#' @return fraction_of_correct_predictions
#' @examples fraction_of_correct_predictions <- calculate_fraction_of_correct_predictions_for_LR_model(LR_model = Direction_vs_Lag2, data_set = Weekly_from_2009_to_2010_inclusive)

#' @export
calculate_fraction_of_correct_predictions_for_LR_model <- function(LR_model, data_set) {
 vector_of_predicted_probabilities <- predict(
  object = LR_model,
  newdata = data_set,
  type = "response"
 )
 number_of_observations <- nrow(data_set)
 vector_of_predicted_directions <- rep("Down", number_of_observations)
 condition <- vector_of_predicted_probabilities > 0.5
 vector_of_predicted_directions[condition] = "Up"
 name_of_response <- names(LR_model$model)[1]
 confusion_matrix <- table(
  vector_of_predicted_directions,
  data_set[, name_of_response]
 )
 number_of_true_negatives <- confusion_matrix[1, 1]
 number_of_false_negatives <- confusion_matrix[1, 2]
 number_of_false_positives <- confusion_matrix[2, 1]
 number_of_true_positives <- confusion_matrix[2, 2]
 number_of_correct_predictions <-
  number_of_true_negatives + number_of_true_positives
 fraction_of_correct_predictions <-
  number_of_correct_predictions / number_of_observations
 return(fraction_of_correct_predictions)
}
