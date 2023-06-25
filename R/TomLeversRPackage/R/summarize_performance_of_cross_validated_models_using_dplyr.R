#' @title summarize_performance_of_cross_validated_models_using_dplyr
#' @description Summarizes performance of cross validated models using dplyr
#' @param formula The formula of the logistic regression model for which to summarize performance of cross validated models using dplyr
#' @param data_frame The data frame on which to train the logistic regression for which to summarize performance of cross validated models using dplyr
#' @return summary_of_performance_of_cross_validated_models_using_dplyr The summary of performance of cross-validated models using dplyr
#' @examples summary_of_performance_of_cross_validated_models_using_dplyr <- summarize_performance_of_cross_validated_models_using_dplyr(Indicator ~ Red + Green + Blue, data_frame = data_frame_of_indicators_and_pixels)
#' @import dplyr
#' @import ggplot2
#' @import rsample

#' @export
summarize_performance_of_cross_validated_models_using_dplyr <- function(type_of_model, formula, data_frame) {
 formula_string <- format(formula)
 print("Summary for model")
 print(paste("of type ", type_of_model, sep = ""))
 print(paste("with formula ", formula_string, sep = ""))
 if (type_of_model == "Logistic Ridge Regression") {
  full_model_matrix <- model.matrix(object = formula, data = data_frame)[, -1]
  full_vector_of_indicators <- data_frame$Indicator
  the_cv.glmnet <- glmnet::cv.glmnet(x = full_model_matrix, y = full_vector_of_indicators, alpha = 0, family = "binomial")
  print(paste("lambda = ", the_cv.glmnet$lambda.min, sep = ""))
 } else if (type_of_model == "KNN") {
  the_trainControl <- caret::trainControl(method  = "cv", number = 10)
  list_of_training_information <- caret::train(
   formula,
   method = "knn",
   tuneGrid = expand.grid(k = 1:nrow(data_frame)),
   trControl = the_trainControl,
   metric = "Accuracy",
   data = data_frame
  )
  K = list_of_training_information$bestTune
  print(paste("K = ", K, sep = ""))
 }
 generate_data_frame_of_predicted_probabilities_and_indicators <-
  function(train_test_split) {
   training_data <- analysis(x = train_test_split)
   testing_data <- assessment(x = train_test_split)
   if (type_of_model == "Logistic Regression") {
    logistic_regression_model <- glm(
     formula = formula,
     data = training_data,
     family = binomial
    )
    vector_of_predicted_probabilities <- predict(
     object = logistic_regression_model,
     newdata = testing_data,
     type = "response"
    )
   } else if (type_of_model == "Logistic Ridge Regression") {
    training_model_matrix <- model.matrix(object = formula, data = training_data)[, -1]
    training_vector_of_indicators <- training_data$Indicator
    logistic_regression_with_lasso_model <- glmnet::glmnet(x = training_model_matrix, y = training_vector_of_indicators, alpha = 1, family = "binomial", lambda = the_cv.glmnet$lambda.min)
    testing_model_matrix <- model.matrix(object = formula, data = testing_data)[, -1]
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
    names_of_variables <- all.vars(formula)
    name_of_response <- names_of_variables[1]
    vector_of_names_of_predictors <- names_of_variables[-1]
    matrix_of_values_of_predictors_for_training <- as.matrix(x = training_data[, vector_of_names_of_predictors])
    matrix_of_values_of_predictors_for_testing <- as.matrix(x = testing_data[, vector_of_names_of_predictors])
    vector_of_response_values_for_training <- training_data[, name_of_response]

    data_frame_of_predicted_probabilities <- predict(
     object = caret::knn3(
      matrix_of_values_of_predictors_for_training,
      vector_of_response_values_for_training,
      k = K
     ),
     matrix_of_values_of_predictors_for_testing
    )
    index_of_column_1 <- get_index_of_column_of_data_frame(data_frame_of_predicted_probabilities, 1)
    vector_of_predicted_probabilities <- data_frame_of_predicted_probabilities[, index_of_column_1]
   } else {
    error_message <- paste("The performance of models of type ", type_of_model, " cannot be yet summarized.", sep = "")
    stop(error_message)
   }
   data_frame_of_predicted_probabilities_and_indicators <- data.frame(
    actual_indicator = testing_data$Indicator,
    predicted_probability = vector_of_predicted_probabilities
   )
   colnames(data_frame_of_predicted_probabilities_and_indicators) <- c("actual_indicator", "predicted_probability")
   return(data_frame_of_predicted_probabilities_and_indicators)
  }
 data_frame_of_sensitivities_and_FPRs <-
  rsample::vfold_cv(data_frame, v = 10, repeats = 1) %>%
  mutate(
   predicted_probability = purrr::map(
    splits,
    generate_data_frame_of_predicted_probabilities_and_indicators
   )
  ) %>%
  tidyr::unnest(predicted_probability) %>%
  group_by(id) %>%
  summarise(
   threshold = provide_performance_metrics(actual_indicator, predicted_probability)$threshold,
   precision = provide_performance_metrics(actual_indicator, predicted_probability)$PPV,
   recall = provide_performance_metrics(actual_indicator, predicted_probability)$TPR,
   F1_measure = (1 + 1^2) * precision * recall / (1^2 * precision + recall),
   decimal_of_true_positives = provide_performance_metrics(actual_indicator, predicted_probability)$decimal_of_true_positives,
   range_of_numbers_of_observations = 1:length(recall)
  )
 data_frame_of_average_sensitivities_and_FPRs <-
  data_frame_of_sensitivities_and_FPRs %>%
  ungroup %>%
  group_by(range_of_numbers_of_observations) %>%
  summarise(
   threshold = mean(threshold),
   precision = mean(precision),
   recall = mean(recall),
   F1_measure = mean(F1_measure),
   decimal_of_true_positives = mean(decimal_of_true_positives),
   id = "Average"
  )
 #mean_AUC <- Bolstad2::sintegral(data_frame_of_average_sensitivities_and_FPRs$recall, data_frame_of_average_sensitivities_and_FPRs$precision)$int
 #data_frame_of_sensitivities_and_FPRs <-
 # bind_rows(
 #  data_frame_of_sensitivities_and_FPRs,
 #  data_frame_of_average_sensitivities_and_FPRs
 # ) %>%
 # mutate(
 #  colour = factor(
 #   ifelse(
 #    test = id == "Average",
 #    yes = "Average",
 #    no = "Individual"
 #   ),
 #   levels = c(
 #    "Individual",
 #    "Average"
 #   )
 #  )
 # )
 ROC_curve <- ggplot(
  data = data_frame_of_average_sensitivities_and_FPRs,
  mapping = aes(x = threshold)
 ) +
  geom_line(mapping = aes(y = decimal_of_true_positives, color = "Average Decimal Of True Predictions")) +
  geom_line(mapping = aes(y = F1_measure, color = "Average F1 measure")) +
  geom_line(mapping = aes(y = precision, color = "Average Precision")) +
  geom_line(mapping = aes(y = recall, color = "Average Recall")) +
  scale_colour_manual(values = c("#008080", "green4", "purple", "red")) +
  theme(legend.position = c(0.5, 0.25)) +
  labs(x = "threshold", y = "performance metric")
 maximum_average_F1_measure <- max(data_frame_of_average_sensitivities_and_FPRs$F1_measure, na.rm = TRUE)
 index_of_column_F1_measure <- get_index_of_column_of_data_frame(data_frame_of_average_sensitivities_and_FPRs, "F1_measure")
 index_of_maximum_average_F1_measure <- which(data_frame_of_average_sensitivities_and_FPRs[, index_of_column_F1_measure] == maximum_average_F1_measure)
 ROC_curve_and_mean_AUC <- list(
  ROC_curve = ROC_curve,
  data_frame_corresponding_to_maximum_average_F1_measure = data_frame_of_average_sensitivities_and_FPRs[index_of_maximum_average_F1_measure, c("threshold", "decimal_of_true_positives", "precision", "recall", "F1_measure")]
 )
 return(ROC_curve_and_mean_AUC)
}
