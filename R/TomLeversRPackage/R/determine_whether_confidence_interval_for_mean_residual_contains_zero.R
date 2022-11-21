determine_whether_confidence_interval_for_mean_residual_contains_zero <- function(linear_model) {
    residual <- linear_model$residuals
    predicted_value <- linear_model$fitted.values
    mean_predicted_value <- mean(predicted_value)
    data_frame_of_residual_and_predicted_value <- data.frame(residual, predicted_value)
    linear_model_of_residual_vs_predicted_value <- lm(residual ~ predicted_value, data = data_frame_of_residual_and_predicted_value)
    data_frame_encapsulating_mean_predicted_value <- data.frame(predicted_value = mean_predicted_value)
    significance_level <- 0.05
    confidence_level <- 1 - significance_level
    confidence_interval_for_mean_residual <- predict(linear_model_of_residual_vs_predicted_value, data_frame_encapsulating_mean_predicted_value, level = confidence_level, interval="confidence")
    lower_bound <- confidence_interval_for_mean_residual[1, "lwr"]
    upper_bound <- confidence_interval_for_mean_residual[1, "upr"]
    if (lower_bound <= 0 && upper_bound >= 0) {
        return(TRUE)
    } else {
        return(FALSE)
    }
}
