test_that("MLR works",  {
    delivery_time_data_frame <- read.csv("Delivery_Time_Data.csv", header = TRUE)
    data_frame_of_delivery_times <- delivery_time_data_frame[1]
    vector_of_delivery_times <- data.matrix(data_frame_of_delivery_times)
    data_frame_of_numbers_of_cases_and_distances <- delivery_time_data_frame[,2:3]
    number_of_observations <- nrow(delivery_time_data_frame)
    vector_of_ones <- rep(1, number_of_observations)
    data_frame_of_ones_number_of_cases_and_distances <- cbind(vector_of_ones, data_frame_of_numbers_of_cases_and_distances)
    model_matrix <- data.matrix(data_frame_of_ones_number_of_cases_and_distances)
    transpose_of_model_matrix_times_model_matrix <- t(model_matrix) %*% model_matrix
    inverse_of_transpose_of_model_matrix_times_model_matrix <- solve(transpose_of_model_matrix_times_model_matrix)
    transpose_of_model_matrix_times_vector_of_delivery_times <- t(model_matrix) %*% vector_of_delivery_times
    vector_of_estimated_multiple_linear_regression_model_coefficients <- inverse_of_transpose_of_model_matrix_times_model_matrix %*% transpose_of_model_matrix_times_vector_of_delivery_times
    linear_model <- lm(y ~ ., data = delivery_time_data_frame)
    vector_of_estimated_multiple_linear_regression_model_coefficients <- linear_model$coefficients
    summary <- summarize_linear_model(linear_model)
    vector_of_delivery_times <- linear_model$model[,1]
    vector_of_fitted_values <- linear_model$fitted.values
    vector_of_residuals <- linear_model$residuals
    analysis <- analyze_variance_for_one_linear_model(linear_model)
    #print(summary)
    #print(analysis)
})
