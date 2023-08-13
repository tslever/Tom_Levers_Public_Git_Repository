#' @export
generate_data_frame_of_actual_indicators_and_predicted_probabilities <- function(formula, testing_data, training_data, type_of_model, list_of_hyperparameters) {
 if (type_of_model == "Logistic Regression") {
  logistic_regression_classifier <- glm(
   formula = formula,
   data = training_data,
   family = binomial
  )
  #if (!file.exists("logistic_regression_classifier.rds")) {
  # saveRDS(logistic_regression_classifier, "logistic_regression_classifier.rds")
  #}
  vector_of_predicted_probabilities <- predict(
   object = logistic_regression_classifier,
   newdata = testing_data,
   type = "response"
  )
 } else if (type_of_model == "Logistic Ridge Regression") {
  training_model_matrix <- model.matrix(object = formula, data = training_data)[, -1]
  training_vector_of_indicators <- training_data$Indicator
  if (number_of_predictors == 1) {
   training_model_matrix <- cbind(0, training_model_matrix)
  }
  lambda <- list_of_hyperparameters[["lambda"]]
  logistic_regression_with_lasso_model <- glmnet::glmnet(x = training_model_matrix, y = training_vector_of_indicators, alpha = 0, family = "binomial", lambda = lambda)
  testing_model_matrix <- model.matrix(object = formula, data = testing_data)[, -1]
  if (number_of_predictors == 1) {
   testing_model_matrix <- cbind(0, testing_model_matrix)
  }
  vector_of_predicted_probabilities <- predict(object = logistic_regression_with_lasso_model, newx = testing_model_matrix, type = "response")
 } else if (type_of_model == "LDA" | type_of_model == "QDA") {
  if (type_of_model == "LDA") {
   model <- MASS::lda(
    formula = formula,
    data = training_data
   )
  } else if (type_of_model == "QDA") {
   model <- MASS::qda(
    formula = formula,
    data = training_data
   )
  }
  prediction <- predict(model, newdata = testing_data)
  data_frame_of_predicted_probabilities <- prediction$posterior
  index_of_column_1 <- get_index_of_column_of_data_frame(data_frame_of_predicted_probabilities, 1)
  vector_of_predicted_probabilities <- data_frame_of_predicted_probabilities[, index_of_column_1]
 } else if (type_of_model == "KNN") {
  matrix_of_values_of_predictors_for_training <- as.matrix(x = training_data[, vector_of_names_of_predictors])
  matrix_of_values_of_predictors_for_testing <- as.matrix(x = testing_data[, vector_of_names_of_predictors])
  vector_of_response_values_for_training <- training_data[, name_of_response]
  the_knn3 <- caret::knn3(
   matrix_of_values_of_predictors_for_training,
   vector_of_response_values_for_training,
   k = optimal_K
  )
  data_frame_of_predicted_probabilities <- predict(
   object = the_knn3,
   matrix_of_values_of_predictors_for_testing
  )
  index_of_column_1 <- get_index_of_column_of_data_frame(data_frame_of_predicted_probabilities, 1)
  vector_of_predicted_probabilities <- data_frame_of_predicted_probabilities[, index_of_column_1]
 } else if (type_of_model == "Random Forest") {
  index_of_column_Indicator <- get_index_of_column_of_data_frame(training_data, "Indicator")
  data_frame_of_training_predictors <- training_data[, -index_of_column_Indicator]
  data_frame_of_training_response_values <- training_data[, index_of_column_Indicator]
  the_randomForest <- randomForest::randomForest(
   formula,
   training_data,
   mtry = optimal_mtry,
   ntree = optimal_number_of_trees
  )
  matrix_of_predicted_probabilities <- predict(the_randomForest, newdata = testing_data, type = "prob")
  vector_of_predicted_probabilities <- matrix_of_predicted_probabilities[, 2]
 } else if (startsWith(type_of_model, "Support-Vector Machine")) {
  SVM <- NULL
  if (type_of_model == "Support-Vector Machine With Linear Kernel") {
   SVM <- e1071::svm(
    formula,
    data = training_data,
    kernel = "linear",
    cost = optimal_cost,
    probability = TRUE
   )
  } else if (type_of_model == "Support-Vector Machine With Polynomial Kernel") {
   SVM <- e1071::svm(
    formula,
    data = training_data,
    kernel = "polynomial",
    cost = optimal_cost,
    degree = optimal_degree,
    probability = TRUE
   )
  } else if (type_of_model == "Support-Vector Machine With Radial Kernel") {
   SVM <- e1071::svm(
    formula,
    data = training_data,
    kernel = "radial",
    cost = optimal_cost,
    gamma = optimal_gamma,
    probability = TRUE
   )
  } else {
   error_message <- paste("A model of type ", type_of_model, " cannot be yet generated.", sep = "")
   stop(error_message)
  }
  factor_of_predictions_and_predicted_probabilities <- predict(SVM, newdata = testing_data, probability = TRUE)
  matrix_of_predicted_probabilities <- attr(x = factor_of_predictions_and_predicted_probabilities, which = "probabilities")
  vector_of_predicted_probabilities <- matrix_of_predicted_probabilities[, 2]
 } else {
  error_message <- paste("The performance of models of type ", type_of_model, " cannot be yet summarized.", sep = "")
  stop(error_message)
 }
 data_frame_of_actual_indicators_and_predicted_probabilities <- data.frame(
  actual_indicator = testing_data$Indicator,
  predicted_probability = vector_of_predicted_probabilities
 )
 colnames(data_frame_of_actual_indicators_and_predicted_probabilities) <- c("actual_indicator", "predicted_probability")
 return(data_frame_of_actual_indicators_and_predicted_probabilities)
}
