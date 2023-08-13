#' @export
visualize_performance_metrics <- function(formula, testing_data, training_data, type_of_model, list_of_hyperparameters) {
 data_frame_of_actual_indicators_and_predicted_probabilities <- generate_data_frame_of_actual_indicators_and_predicted_probabilities(formula, testing_data, training_data, type_of_model, list_of_hyperparameters)
 data_frame_of_performance_metrics <- data_frame_of_actual_indicators_and_predicted_probabilities %>% reframe(
  accuracy = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "accuracy"),
  decimal_of_true_positives = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "decimal of true positives"),
  F1_measure = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "F1 measure"),
  FPR = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "FPR"),
  number_of_negatives = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "number of negatives"),
  number_of_observations = 1:length(accuracy),
  number_of_positives = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "number of positives"),
  PPV = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "PPV"),
  threshold = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "threshold"),
  TPR = provide_vector_of_values_of_performance_metric(actual_indicator, predicted_probability, "TPR"),
 )
 plot_of_performance_metrics_vs_threshold <- ggplot(
  data = data_frame_of_performance_metrics,
  mapping = aes(x = threshold)
 ) +
  geom_line(mapping = aes(y = accuracy, color = "Average Accuracy")) +
  geom_line(mapping = aes(y = decimal_of_true_positives, color = "Average Decimal Of True Predictions")) +
  geom_line(mapping = aes(y = F1_measure, color = "Average F1 measure")) +
  geom_line(mapping = aes(y = PPV, color = "Average PPV")) +
  geom_line(mapping = aes(y = TPR, color = "Average TPR")) +
  scale_colour_manual(values = c("red", "orange", "yellow", "green", "blue")) +
  scale_x_continuous(breaks = seq(from = 0, to = 1, by = 0.05)) +
  theme(legend.position = c(0.5, 0.5)) +
  labs(x = "threshold", y = "performance metric", title = "Average Performance Metrics Vs. Threshold") +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
  )
 data_frame_of_PPV_and_TPR <- data_frame_of_performance_metrics[, c("PPV", "TPR")]
 number_of_thresholds <- nrow(data_frame_of_PPV_and_TPR)
 number_of_negatives <- data_frame_of_performance_metrics[1, "number_of_negatives"]
 number_of_positives <- data_frame_of_performance_metrics[1, "number_of_positives"]
 data_frame_of_PPV_and_TPR[number_of_thresholds + 1, "PPV"] <- 1
 data_frame_of_PPV_and_TPR[number_of_thresholds + 1, "TPR"] <- 0
 number_of_thresholds <- nrow(data_frame_of_PPV_and_TPR)
 data_frame_of_PPV_and_TPR[number_of_thresholds + 1, "PPV"] <- number_of_positives / (number_of_negatives + number_of_positives)
 data_frame_of_PPV_and_TPR[number_of_thresholds + 1, "TPR"] <- 1
 PR_curve <- ggplot(
  data = data_frame_of_PPV_and_TPR,
  mapping = aes(x = TPR)
 ) +
  geom_line(mapping = aes(y = PPV)) +
  labs(x = "TPR", y = "PPV", title = "PPV-TPR Curve") +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
  )
 data_frame_of_TPR_and_FPR <- data_frame_of_performance_metrics[, c("TPR", "FPR")]
 number_of_thresholds <- nrow(data_frame_of_TPR_and_FPR)
 data_frame_of_TPR_and_FPR[number_of_thresholds + 1, "TPR"] <- 0
 data_frame_of_TPR_and_FPR[number_of_thresholds + 1, "FPR"] <- 0
 number_of_thresholds <- nrow(data_frame_of_TPR_and_FPR)
 data_frame_of_TPR_and_FPR[number_of_thresholds + 1, "TPR"] <- 1
 data_frame_of_TPR_and_FPR[number_of_thresholds + 1, "FPR"] <- 1
 ROC_curve <- ggplot(
  data = data_frame_of_TPR_and_FPR,
  mapping = aes(x = FPR)
 ) +
  geom_line(mapping = aes(y = TPR)) +
  labs(x = "FPR", y = "TPR", title = "ROC Curve And TPR Vs. FPR") +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
  )
 maximum_F1_measure <- max(data_frame_of_performance_metrics$F1_measure, na.rm = TRUE)
 index_of_column_F1_measure <- get_index_of_column_of_data_frame(data_frame_of_performance_metrics, "F1_measure")
 index_of_maximum_F1_measure <- which(data_frame_of_performance_metrics[, index_of_column_F1_measure] == maximum_F1_measure)
 data_frame_corresponding_to_maximum_F1_measure <-
  data_frame_of_performance_metrics[index_of_maximum_F1_measure, c("threshold", "accuracy", "TPR", "FPR", "PPV", "F1_measure")] %>%
  rename(corresponding_threshold = threshold, corresponding_accuracy = accuracy, corresponding_TPR = TPR, corresponding_FPR = FPR, corresponding_PPV = PPV, optimal_F1_measure = F1_measure)
 data_frame_corresponding_to_maximum_F1_measure <- data_frame_corresponding_to_maximum_F1_measure[1, ]
 optimal_F1_measure <- data_frame_corresponding_to_maximum_F1_measure$optimal_F1_measure
 data_frame_of_optimal_performance_metrics <- data.frame(
  corresponding_threshold = data_frame_corresponding_to_maximum_F1_measure$corresponding_threshold,
  optimal_PR_AUC = MESS::auc(data_frame_of_PPV_and_TPR$TPR, data_frame_of_PPV_and_TPR$PPV),
  optimal_ROC_AUC = MESS::auc(data_frame_of_TPR_and_FPR$FPR, data_frame_of_TPR_and_FPR$TPR),
  corresponding_accuracy = data_frame_corresponding_to_maximum_F1_measure$corresponding_accuracy,
  corresponding_TPR = data_frame_corresponding_to_maximum_F1_measure$corresponding_TPR,
  corresponding_FPR = data_frame_corresponding_to_maximum_F1_measure$corresponding_FPR,
  corresponding_PPV = data_frame_corresponding_to_maximum_F1_measure$corresponding_PPV,
  optimal_F1_measure = optimal_F1_measure
 )
 list_of_visualizations_of_performance_metrics <- list(
  plot_of_performance_metrics_vs_threshold = plot_of_performance_metrics_vs_threshold,
  PR_curve = PR_curve,
  ROC_curve = ROC_curve,
  data_frame_of_optimal_performance_metrics = data_frame_of_optimal_performance_metrics
 )
 return(list_of_visualizations_of_performance_metrics)
}
