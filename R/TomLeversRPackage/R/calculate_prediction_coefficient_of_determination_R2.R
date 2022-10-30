calculate_prediction_coefficient_of_determination_R2 <- function(linear_model) {
    PRESS <- calculate_PRESS(linear_model)
    total_sum_of_squares <- calculate_total_sum_of_squares(linear_model)
    prediction_coefficient_of_determination_R2 <- 1 - (PRESS / total_sum_of_squares)
}
