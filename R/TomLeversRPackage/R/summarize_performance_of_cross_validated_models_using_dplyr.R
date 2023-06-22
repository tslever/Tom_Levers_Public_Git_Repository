#' @title summarize_performance_of_cross_validated_models_using_dplyr
#' @description Summarizes performance of cross validated models using dplyr
#' @param formula The formula of the logistic regression model for which to summarize performance of cross validated models using dplyr
#' @param data_frame The data frame on which to train the logistic regression for which to summarize performance of cross validated models using dplyr
#' @return summary_of_performance_of_cross_validated_models_using_dplyr The summary of performance of cross-validated models using dplyr
#' @examples summary_of_performance_of_cross_validated_models_using_dplyr <- summarize_performance_of_cross_validated_models_using_dplyr(Indicator ~ Red + Green + Blue, data_frame = data_frame_of_indicators_and_pixels)
#' @import dplyr
#' @import rsample

#' @export
summarize_performance_of_cross_validated_models_using_dplyr <- function(type_of_model, formula, data_frame, K = 1) {
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
    data_frame_of_predicted_probabilities <- predict(caret::knn3(matrix_of_values_of_predictors_for_training, vector_of_response_values_for_training, k = K), matrix_of_values_of_predictors_for_testing)
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
   precision = provide_data_for_ROC_and_PR_curves(actual_indicator, predicted_probability)$PPV,
   recall = provide_data_for_ROC_and_PR_curves(actual_indicator, predicted_probability)$TPR,
   range_of_numbers_of_observations = 1:length(recall)
  )
 data_frame_of_average_sensitivities_and_FPRs <-
  data_frame_of_sensitivities_and_FPRs %>%
  ungroup %>%
  group_by(range_of_numbers_of_observations) %>%
  summarise(
   precision = mean(precision),
   recall = mean(recall),
   id = "Average"
  )
 mean_AUC <- Bolstad2::sintegral(data_frame_of_average_sensitivities_and_FPRs$recall, data_frame_of_average_sensitivities_and_FPRs$precision)$int
 data_frame_of_sensitivities_and_FPRs <-
  bind_rows(
   data_frame_of_sensitivities_and_FPRs,
   data_frame_of_average_sensitivities_and_FPRs
  ) %>%
  mutate(
   colour = factor(
    ifelse(
     test = id == "Average",
     yes = "Average",
     no = "Individual"
    ),
    levels = c(
     "Individual",
     "Average"
    )
   )
  )
 library(ggplot2)
 ROC_curve <- ggplot(
  data = data_frame_of_sensitivities_and_FPRs,
  mapping = aes(x = recall, y = precision, group = id, colour = colour)
 ) +
  geom_line(mapping = aes(size = colour, alpha = colour)) +
  scale_colour_manual(values = c("black", "red")) +
  scale_size_manual(values = c(0.5, 1.25)) +
  scale_alpha_manual(values = c(0.3, 1)) +
  theme_classic() +
  theme(legend.position = c(0.75, 0.15)) +
  labs(x = "recall", y = "precision", colour = "", alpha = "", size = "")
 ROC_curve_and_mean_AUC <- list(
  ROC_curve = ROC_curve,
  mean_AUC = mean_AUC
 )
 return(ROC_curve_and_mean_AUC)
}
