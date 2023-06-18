#' @title summarize_performance_of_cross_validated_models_using_dplyr
#' @description Summarizes performance of cross validated models using dplyr
#' @param formula The formula of the logistic regression model for which to summarize performance of cross validated models using dplyr
#' @param data_frame The data frame on which to train the logistic regression for which to summarize performance of cross validated models using dplyr
#' @return summary_of_performance_of_cross_validated_models_using_dplyr The summary of performance of cross-validated models using dplyr
#' @examples summary_of_performance_of_cross_validated_models_using_dplyr <- summarize_performance_of_cross_validated_models_using_dplyr(Indicator ~ Red + Green + Blue, data_frame = data_frame_of_indicators_and_pixels)
#' @import dplyr
#' @import rsample

#' @export
summarize_performance_of_cross_validated_models_using_dplyr <- function(formula, data_frame) {
 generate_data_frame_of_predicted_probabilities_and_indicators <-
  function(train_test_split) {
   formula <- formula
   training_data <- analysis(x = train_test_split)
   logistic_regression_model <- glm(
    formula = formula,
    data = training_data,
    family = binomial
   )
   testing_data <- assessment(x = train_test_split)
   vector_of_predicted_probabilities <- predict(
    object = logistic_regression_model,
    newdata = testing_data,
    type = "response"
   )
   data_frame_with_variables_needed_to_use_formula <- model.frame(
    formula = formula,
    data = testing_data
   )
   vector_of_predicted_indicators_of_whether_murder_rate_is_higher_than_mean <-
    model.response(data = data_frame_with_variables_needed_to_use_formula)
   data_frame_of_predicted_probabilities_and_indicators <- data.frame(
    predicted_probability = vector_of_predicted_probabilities,
    predicted_indicator =
     vector_of_predicted_indicators_of_whether_murder_rate_is_higher_than_mean
   )
   return(data_frame_of_predicted_probabilities_and_indicators)
  }
 data_frame_of_sensitivities_and_specificities <-
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
   sensitivity = pROC::roc(
    predicted_indicator,
    predicted_probability,
    plot = FALSE
   )$sensitivities,
   specificity = pROC::roc(
    predicted_indicator,
    predicted_probability,
    plot = FALSE
   )$specificities,
   range_of_numbers_of_observations = 1:length(sensitivity)
  )
 data_frame_of_average_sensitivities_and_specificities <-
  data_frame_of_sensitivities_and_specificities %>%
  ungroup %>%
  group_by(range_of_numbers_of_observations) %>%
  summarise(
   sensitivity = mean(sensitivity),
   specificity = mean(specificity),
   id = "Average"
  )
 data_frame_of_sensitivities_and_specificities <-
  bind_rows(
   data_frame_of_sensitivities_and_specificities,
   data_frame_of_average_sensitivities_and_specificities
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
  data = data_frame_of_sensitivities_and_specificities,
  mapping = aes(x = 1 - sensitivity, y = specificity, group = id, colour = colour)
 ) +
  geom_line(mapping = aes(size = colour, alpha = colour)) +
  scale_colour_manual(values = c("black", "red")) +
  scale_size_manual(values = c(0.5, 1.25)) +
  scale_alpha_manual(values = c(0.3, 1)) +
  theme_classic() +
  theme(legend.position = c(0.75, 0.15)) +
  labs(x = "1 - Sensitivity", y = "Specificity", colour = "", alpha = "", size = "")
 data_frame_of_id_and_AUC <- rsample::vfold_cv(
  data = data_frame,
  v = 10,
  repeats = 1
 ) %>%
  mutate(
   predicted_probability = purrr::map(
    splits,
    generate_data_frame_of_predicted_probabilities_and_indicators
   )
  ) %>%
  tidyr::unnest(predicted_probability) %>%
  group_by(id) %>%
  summarise(
   AUC = pROC::roc(
    predicted_indicator,
    predicted_probability,
    plot = FALSE
   )$auc[1]
  )
 mean_AUC <- mean(data_frame_of_id_and_AUC$AUC)
 ROC_curve_and_mean_AUC <- list(
  ROC_curve = ROC_curve,
  mean_AUC = mean_AUC
 )
 return(ROC_curve_and_mean_AUC)
}
