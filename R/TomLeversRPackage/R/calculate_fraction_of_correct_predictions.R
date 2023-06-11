#' @title calculate_fraction_of_correct_predictions
#' @description Calculates the fraction of correct predictions for a model given a data set for which to predict
#' @param model The model for which to calculate the fraction of correct predictions given a data set for which to predict
#' @return fraction_of_correct_predictions
#' @examples fraction_of_correct_predictions <- calculate_fraction_of_correct_predictions(model = Direction_vs_Lag2, data_set = Weekly_from_2009_to_2010_inclusive)

#' @export
calculate_fraction_of_correct_predictions <- function(type_of_model, formula, training_data, test_data) {
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
 } else {
  stop("A fraction of correct predictions may not be yet calculated for models of the class of parameter model.")
 }
 confusion_matrix <- table(
  vector_of_predicted_directions,
  test_data[, name_of_response]
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
