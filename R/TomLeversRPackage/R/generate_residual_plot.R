#' @export
generate_residual_plot <- function(linear_model, should_plot_residuals_versus_predicted_values) {
 if (should_plot_residuals_versus_predicted_values) {
     dependent_variable_values <- linear_model$fitted.values
     plot_title <- "Residual vs. Predicted Value"
 } else {
     dependent_variable_values <- linear_model$model[, 2]
     plot_title <- "Residual vs. Predictor Value"
 }
 residual_plot <- ggplot(
  data.frame(
   residual = linear_model$residuals,
   dependent_variable_value = dependent_variable_values
  ),
  aes(x = dependent_variable_value, y = residual)
 ) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, color = "red") +
  labs(
   x = "predicted value",
   y = "residual",
   title = plot_title
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
 return(residual_plot)
}
