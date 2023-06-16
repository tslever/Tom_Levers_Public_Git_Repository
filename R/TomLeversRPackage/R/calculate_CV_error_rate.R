#' @title calculate_LOOCV_error_rate
#' @description Calculates LOOCV error rate
#' @param type_of_model The type of model. An element in the set {"Logistic Regression"}.
#' @return LOOCV_error_rate The error rate of Leave One Out Cross Validation
#' @examples LOOCV_error_rate <- calculate_LOOCV_error_rate(type_of_model = "LR", formula = Direction ~ Lag1 + Lag2, data_frame = ISLR2::Weekly)

#' @export
calculate_CV_error_rate <- function(type_of_model, formula, data_frame, number_of_folds) {
 number_of_observations <- nrow(data_frame)
 number_of_data_per_fold <- ceiling(number_of_observations / number_of_folds)
 if (type_of_model == "Polynomial Regression") {
  vector_of_mean_squared_errors <- double(0)
  for (i in 1:number_of_folds) {
   if (number_of_folds == 1) {
    training_data <- data_frame
    testing_data <- data_frame
   } else {
    if (i < number_of_folds) {
     vector_of_indices_of_observations_in_fold <- c((1+number_of_data_per_fold*(i-1)):(number_of_data_per_fold*i))
     training_data <- data_frame[-vector_of_indices_of_observations_in_fold, ]
     testing_data <- data_frame[vector_of_indices_of_observations_in_fold, ]
    } else {
     #i <- number_of_folds
     vector_of_indices_of_observations_in_fold <- c((1+number_of_data_per_fold*(i-1)):number_of_observations)
     training_data <- data_frame[-vector_of_indices_of_observations_in_fold, ]
     testing_data <- data_frame[vector_of_indices_of_observations_in_fold, ]
    }
   }
   linear_regression_model <- glm(
    formula = formula,
    data = training_data
   )
   mean_squared_error <- calculate_mean_squared_error(linear_regression_model)
   vector_of_mean_squared_errors <- append(vector_of_mean_squared_errors, mean_squared_error)
  }
  LOOCV_error_rate <- mean(vector_of_mean_squared_errors)
  return(LOOCV_error_rate)
 } else if (type_of_model == "Logistic Regression") {
  vector_of_indicators_of_error <- integer(0)
  for (i in 1:number_of_folds) {
   if (number_of_folds == 1) {
    training_data <- data_frame
    testing_data <- data_frame
   } else {
    if (i < number_of_folds) {
     vector_of_indices_of_observations_in_fold <- c((1+number_of_data_per_fold*(i-1)):(number_of_data_per_fold*i))
     training_data <- data_frame[-vector_of_indices_of_observations_in_fold, ]
     testing_data <- data_frame[vector_of_indices_of_observations_in_fold, ]
    } else {
     #i <- number_of_folds
     vector_of_indices_of_observations_in_fold <- c((1+number_of_data_per_fold*(i-1)):number_of_observations)
     training_data <- data_frame[-vector_of_indices_of_observations_in_fold, ]
     testing_data <- data_frame[vector_of_indices_of_observations_in_fold, ]
    }
   }
   LR_model <- glm(
    formula = formula,
    data = training_data,
    family = binomial
   )
   vector_of_predicted_probabilities <- predict(
    object = LR_model,
    newdata = testing_data,
    type = "response"
   )
   name_of_response <- names(LR_model$model)[1]
   factor_of_response_values <- testing_data[, name_of_response]
   vector_of_levels <- attr(factor_of_response_values, "levels")
   lower_level <- vector_of_levels[1]
   upper_level <- vector_of_levels[2]
   number_of_test_observations <- nrow(testing_data)
   vector_of_predicted_response_values <- rep(lower_level, number_of_test_observations)
   condition <- vector_of_predicted_probabilities > 0.5
   vector_of_predicted_response_values[condition] <- upper_level
   for (j in 1:number_of_test_observations) {
    if (vector_of_predicted_response_values[j] == testing_data[j, name_of_response]) {
     vector_of_indicators_of_error <- append(vector_of_indicators_of_error, 0)
    } else {
     vector_of_indicators_of_error <- append(vector_of_indicators_of_error, 1)
    }
   }
  }
  LOOCV_error_rate <- mean(vector_of_indicators_of_error)
  return(LOOCV_error_rate)
 } else {
  error_message <- paste("LOOCV error rate may not be calculated yet for type of model ", type_of_model, sep = "")
  stop(error_message)
 }
}
