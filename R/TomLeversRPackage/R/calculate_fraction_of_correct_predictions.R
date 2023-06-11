#' @title calculate_fraction_of_correct_predictions
#' @description Calculates the fraction of correct predictions for a model given a data set for which to predict
#' @param model The model for which to calculate the fraction of correct predictions given a data set for which to predict
#' @return fraction_of_correct_predictions
#' @examples fraction_of_correct_predictions <- calculate_fraction_of_correct_predictions(model = Direction_vs_Lag2, data_set = Weekly_from_2009_to_2010_inclusive)

#' @export
calculate_fraction_of_correct_predictions <- function(model, data_set) {
 vector_of_classes_of_model <- class(model)
 number_of_observations <- nrow(data_set)
 if ("glm" %in% vector_of_classes_of_model) {
  vector_of_predicted_probabilities <- predict(
   object = model,
   newdata = data_set,
   type = "response"
  )
  vector_of_predicted_directions <- rep("Down", number_of_observations)
  condition <- vector_of_predicted_probabilities > 0.5
  vector_of_predicted_directions[condition] = "Up"
  name_of_response <- names(model$model)[1]
 } else if (("lda" %in% vector_of_classes_of_model) | "qda" %in% vector_of_classes_of_model) {
  prediction <- predict(model, newdata = data_set)
  vector_of_predicted_directions <- prediction$class
  name_of_response <- names(attr(model$terms, "dataClasses"))[1]
 } else {
  stop("A fractions of correct predictions may not be yet calculated for models of the class of parameter model.")
 }
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
