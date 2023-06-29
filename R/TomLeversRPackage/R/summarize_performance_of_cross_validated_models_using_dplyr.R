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
 print(paste("Summary for model of type ", type_of_model, sep = ""))
 vector_of_variables <- all.vars(formula)
 print(paste(vector_of_variables[1], " ~ ", sep = ""))
 print(paste("    ", vector_of_variables[2], sep = ""))
 number_of_variables <- length(vector_of_variables)
 if (number_of_variables > 2) {
  for (i in 3:number_of_variables) {
   print(paste("    + ", vector_of_variables[i], sep = ""))
  }
 }
 if (type_of_model == "Logistic Ridge Regression") {
  the_trainControl <- caret::trainControl(method  = "cv", summaryFunction = calculate_F1_measure)
  list_of_training_information <- caret::train(
   form = formula,
   data = data_frame,
   method = "glmnet",
   metric = "F1_measure",
   trControl = the_trainControl,
   tuneGrid = expand.grid(alpha = 0, lambda = seq(from = 0.002, to = 0.004, by = 0.0001))
  )
  optimal_value_of_lambda = list_of_training_information$bestTune$lambda
  print(paste("optimal value of lambda = ", optimal_value_of_lambda, sep = ""))
 } else if (type_of_model == "KNN") {
  the_trainControl <- caret::trainControl(method  = "cv", summaryFunction = calculate_F1_measure)
  list_of_training_information <- caret::train(
   form = formula,
   data = data_frame,
   method = "knn",
   metric = "F1_measure",
   trControl = the_trainControl,
   tuneGrid = expand.grid(k = seq(from = 1, to = 3, by = 1))
  )
  K = list_of_training_information$bestTune
  print(paste("optimal value of K = ", K, sep = ""))
 }
 generate_data_frame_of_actual_indicators_and_predicted_probabilities <-
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
    logistic_regression_with_lasso_model <- glmnet::glmnet(x = training_model_matrix, y = training_vector_of_indicators, alpha = 1, family = "binomial", lambda = optimal_value_of_lambda)
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
    the_knn3 <- caret::knn3(
     matrix_of_values_of_predictors_for_training,
     vector_of_response_values_for_training,
     k = K
    )
    data_frame_of_predicted_probabilities <- predict(
     object = the_knn3,
     matrix_of_values_of_predictors_for_testing
    )
    index_of_column_1 <- get_index_of_column_of_data_frame(data_frame_of_predicted_probabilities, 1)
    vector_of_predicted_probabilities <- data_frame_of_predicted_probabilities[, index_of_column_1]
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
 data_frame_of_performance_metrics <-
  rsample::vfold_cv(data_frame, v = 10, repeats = 1) %>%
  mutate(
   predicted_probability = purrr::map(
    splits,
    generate_data_frame_of_actual_indicators_and_predicted_probabilities
   )
  ) %>%
  tidyr::unnest(predicted_probability) %>%
  group_by(id) %>%
  reframe(
   threshold = provide_performance_metrics(actual_indicator, predicted_probability)$threshold,
   fall_out = provide_performance_metrics(actual_indicator, predicted_probability)$FPR,
   precision = provide_performance_metrics(actual_indicator, predicted_probability)$PPV,
   recall = provide_performance_metrics(actual_indicator, predicted_probability)$TPR,
   F1_measure = (1 + 1^2) * precision * recall / (1^2 * precision + recall),
   decimal_of_true_positives = provide_performance_metrics(actual_indicator, predicted_probability)$decimal_of_true_positives,
   number_of_observations = 1:length(recall)
  )
 data_frame_of_average_performance_metrics <-
  data_frame_of_performance_metrics %>%
  ungroup %>%
  group_by(number_of_observations) %>%
  reframe(
   threshold = mean(threshold),
   fall_out = mean(fall_out),
   precision = mean(precision),
   recall = mean(recall),
   F1_measure = mean(F1_measure),
   decimal_of_true_positives = mean(decimal_of_true_positives),
   id = "Average"
  )
 data_frame_of_average_performance_metrics <- data_frame_of_average_performance_metrics[complete.cases(data_frame_of_average_performance_metrics), ]
 ROC_curve <- ggplot(
  data = data_frame_of_average_performance_metrics,
  mapping = aes(x = fall_out)
 ) +
  geom_line(mapping = aes(y = recall)) +
  labs(x = "fall-out", y = "recall", title = "ROC Curve And Recall Vs. Fall-Out") +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
  )
 data_frame_of_precision_and_recall <- data_frame_of_average_performance_metrics[, c("precision", "recall")]
 number_of_thresholds <- nrow(data_frame_of_precision_and_recall)
 data_frame_of_precision_and_recall[number_of_thresholds + 1, "precision"] = 1
 data_frame_of_precision_and_recall[number_of_thresholds + 1, "recall"] = 0
 PR_curve <- ggplot(
  data = data_frame_of_precision_and_recall,
  mapping = aes(x = recall)
 ) +
  geom_line(mapping = aes(y = precision)) +
  labs(x = "recall", y = "precision", title = "Precision-Recall Curve") +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
  )
 plot_of_performance_metrics_vs_threshold <- ggplot(
  data = data_frame_of_average_performance_metrics,
  mapping = aes(x = threshold)
 ) +
  geom_line(mapping = aes(y = decimal_of_true_positives, color = "Average Decimal Of True Predictions")) +
  geom_line(mapping = aes(y = F1_measure, color = "Average F1 measure")) +
  geom_line(mapping = aes(y = precision, color = "Average Precision")) +
  geom_line(mapping = aes(y = recall, color = "Average Recall")) +
  scale_colour_manual(values = c("#008080", "green4", "purple", "red")) +
  theme(legend.position = c(0.5, 0.25)) +
  labs(x = "threshold", y = "performance metric", title = "Performance Metrics Vs. Threshold") +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
  )
 maximum_average_F1_measure <- max(data_frame_of_average_performance_metrics$F1_measure, na.rm = TRUE)
 index_of_column_F1_measure <- get_index_of_column_of_data_frame(data_frame_of_average_performance_metrics, "F1_measure")
 index_of_maximum_average_F1_measure <- which(data_frame_of_average_performance_metrics[, index_of_column_F1_measure] == maximum_average_F1_measure)
 area_under_PR_curve <- MESS::auc(data_frame_of_precision_and_recall$recall, data_frame_of_precision_and_recall$precision)
 area_under_ROC_curve <- MESS::auc(data_frame_of_average_performance_metrics$fall_out, data_frame_of_average_performance_metrics$recall)
 summary_of_performance_of_cross_validated_models <- list(
  area_under_PR_curve = area_under_PR_curve,
  area_under_ROC_curve = area_under_ROC_curve,
  plot_of_performance_metrics_vs_threshold = plot_of_performance_metrics_vs_threshold,
  PR_curve = PR_curve,
  ROC_curve = ROC_curve,
  data_frame_corresponding_to_maximum_average_F1_measure = data_frame_of_average_performance_metrics[index_of_maximum_average_F1_measure, c("threshold", "decimal_of_true_positives", "fall_out", "precision", "recall", "F1_measure")]
 )
 return(summary_of_performance_of_cross_validated_models)
}

# lev = vector_of_factor_levels_that_correspond_to_results
# model = method_specified_in_train
calculate_F1_measure <- function(data_frame_containing_actual_and_predicted_labels, lev = NULL, model = NULL) {
 F1_measure <- MLmetrics::F1_Score(y_true = data_frame_containing_actual_and_predicted_labels$obs, y_pred = data_frame_containing_actual_and_predicted_labels$pred, positive = lev[1])
 vector_with_F1_measure <- c(F1_measure = F1_measure)
 return(vector_with_F1_measure)
}
