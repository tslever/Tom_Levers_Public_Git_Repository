#' @export
plot_ROC_curve_for_cross_validated_models <- function(type_of_model, formula, data_frame, number_of_folds, step) {
 vector_of_rates_of_false_positives <- double(0)
 vector_of_rates_of_true_positives <- double(0)
 for (i in seq(from = 0, to = 1, by = step)) {
  summary_of_performance_of_cross_validated_models <- summarize_performance_of_cross_validated_models(
   type_of_model,
   formula,
   data_frame,
   number_of_folds,
   threshold = i
  )
  vector_of_rates_of_false_positives <- append(vector_of_rates_of_false_positives, summary_of_performance_of_cross_validated_models$mean_rate_of_false_positives)
  vector_of_rates_of_true_positives <- append(vector_of_rates_of_true_positives, summary_of_performance_of_cross_validated_models$mean_rate_of_true_positives)
 }
 data_frame <- data.frame(
  FPR = vector_of_rates_of_false_positives,
  TPR = vector_of_rates_of_true_positives
 )
 ROC_curve <- ggplot(
  data = data_frame,
  aes(x = FPR, y = TPR)
 ) +
  geom_point(alpha = 0.5) +
  labs(
   x = "FPR",
   y = "TPR",
   title = "ROC Curve"
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
 return(ROC_curve)
}
