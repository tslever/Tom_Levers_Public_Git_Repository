#' @title summarize_all_MLR_models
#' @description Summarizes all MLR models
#' @return The summary of all MLR models
#' @examples summary_of_all_MLR_models <- summarize_all_MLR_models()

#' @export
summarize_all_MLR_models <- function(data_set, response, maximum_number_of_models_in_summary_data_frame_for_each_number_of_variables, serialized_functions_to_apply, Box_Cox_Method_should_be_performed) {
 data_frame_of_all_MLR_models <- data.frame(
  formula_string = character(),
  confidence_interval_for_mean_residual_contains_zero = logical(),
  linear_model_is_homoscedastic = logical(),
  number_of_predictors_in_model = integer(),
  number_of_variables_in_model = integer(),
  residual_sum_of_squares = double(),
  r_squared = double(),
  adjusted_r_squared = double(),
  residual_mean_square = double(),
  mallows_Cp = double(),
  schwartz_bayesian_information_criterion = double(),
  stringsAsFactors = FALSE
 )
 residual_plots <- vector()
 formula_for_full_model <- reformulate(termlabels = ".", response = response)
 subset_selection_object <- regsubsets(formula_for_full_model, data = data_set, nbest = maximum_number_of_models_in_summary_data_frame_for_each_number_of_variables, really.big = TRUE, nvmax = NULL)
 summary_for_subset_selection_object <- summary(subset_selection_object)
 mask_of_predictors_for_all_models <- summary_for_subset_selection_object$which
 names_of_intercept_and_predictors <- colnames(mask_of_predictors_for_all_models)
 names_of_predictors <- names_of_intercept_and_predictors[2:length(names_of_intercept_and_predictors)]
 data_frame_of_all_MLR_models["B_0"] <- double()
 for (name_of_predictor in names_of_predictors) {
     name_of_coefficient <- paste("B_", name_of_predictor, sep = "")
     data_frame_of_all_MLR_models[name_of_coefficient] <- double()
 }
 number_of_possible_MLR_models <- nrow(mask_of_predictors_for_all_models)
 lambdas <- rep(1, times = number_of_possible_MLR_models)
 for (i in 1:number_of_possible_MLR_models) {
  mask_of_predictors <- mask_of_predictors_for_all_models[i,]
  names_of_predictors_in_model <- character()
  for (name_of_predictor in names_of_predictors) {
   predictor_is_in_model <- mask_of_predictors[[name_of_predictor]]
   if (predictor_is_in_model) {
    names_of_predictors_in_model <- append(names_of_predictors_in_model, name_of_predictor)
   }
  }
  number_of_predictors_in_model <- length(names_of_predictors_in_model)
  number_of_variables_in_model <- number_of_predictors_in_model + 1
  response_values <- data_set[, response]
  serialized_function_to_apply <- serialized_functions_to_apply[i, response]
  function_to_apply <- eval(parse(text = serialized_function_to_apply))
  transformed_response_values <- apply(as.array(response_values), 1, function_to_apply)
  transformed_response <- paste("transformed_", response, sep = "")
  formula <- reformulate(termlabels = names_of_predictors_in_model, response = transformed_response)
  transformed_data_set <- data.frame(transformed_response = transformed_response_values)
  colnames(transformed_data_set) <- transformed_response
  for (name_of_predictor_in_model in names_of_predictors_in_model) {
      serialized_function_to_apply <- serialized_functions_to_apply[i, name_of_predictor_in_model]
      function_to_apply <- eval(parse(text = serialized_function_to_apply))
      predictor_values <- data_set[, name_of_predictor_in_model]
      transformed_data_set[name_of_predictor_in_model] <- apply(as.array(predictor_values), 1, function_to_apply)
  }
  linear_model <- lm(formula, data = transformed_data_set)
  residual_sum_of_squares <- calculate_residual_sum_of_squares(linear_model)
  r_squared <- calculate_coefficient_of_determination_R2(linear_model)
  adjusted_r_squared <- calculate_adjusted_coefficient_of_determination_R2(linear_model)
  confidence_interval_for_mean_residual_contains_zero <- determine_whether_confidence_interval_for_mean_residual_contains_zero(linear_model)
  linear_model_is_homoscedastic <- determine_whether_linear_model_is_homoscedastic(linear_model)
  residual_mean_square <- calculate_residual_mean_square(linear_model)
  formula_for_transformed_full_model <- reformulate(termlabels = ".", response = transformed_response)
  transformed_full_model <- lm(formula_for_transformed_full_model, data = transformed_data_set)
  estimate_of_variance_of_errors <- calculate_residual_mean_square(transformed_full_model)
  number_of_data <- nrow(data_set)
  mallows_Cp <- residual_sum_of_squares / estimate_of_variance_of_errors - number_of_data + 2 * number_of_variables_in_model
  schwartz_bayesian_information_criterion <- number_of_data * log(residual_sum_of_squares / number_of_data, base = exp(1)) + number_of_variables_in_model * log(number_of_data, base = exp(1))
  linear_model_coefficients <- linear_model$coefficients
  names_of_linear_model_coefficients <- names(linear_model_coefficients)
  number_of_variables <- length(names_of_intercept_and_predictors)
  coefficients <- double(number_of_variables)
  for (name_of_linear_model_coefficient in names_of_linear_model_coefficients) {
      index_of_name_of_linear_model_coefficient_in_names_of_intercept_and_predictors <- match(name_of_linear_model_coefficient, names_of_intercept_and_predictors)
      coefficients[index_of_name_of_linear_model_coefficient_in_names_of_intercept_and_predictors] = linear_model_coefficients[name_of_linear_model_coefficient]
  }
  formula_as_vector_of_strings <- as.character(formula)
  formula_as_string <- paste(formula_as_vector_of_strings[2], formula_as_vector_of_strings[1], formula_as_vector_of_strings[3], sep = " ")
  data_frame_for_model <- list(
      formula_as_string,
      confidence_interval_for_mean_residual_contains_zero,
      linear_model_is_homoscedastic,
      number_of_predictors_in_model,
      number_of_variables_in_model,
      residual_sum_of_squares,
      r_squared,
      adjusted_r_squared,
      residual_mean_square,
      mallows_Cp,
      schwartz_bayesian_information_criterion
  )
  for (coefficient in coefficients) {
      data_frame_for_model <- append(data_frame_for_model, coefficient)
  }
  data_frame_of_all_MLR_models[nrow(data_frame_of_all_MLR_models) + 1, ] <- data_frame_for_model
  residual_plot <- ggplot(
   data.frame(
    residual = linear_model$residuals,
    predicted_transformed_response = linear_model$fitted.values
   ),
   aes(x = predicted_transformed_response, y = residual)
  ) +
   geom_point(alpha = 0.2) +
   geom_hline(yintercept = 0, color = "red") +
   labs(
    x = paste("predicted ", transformed_response, sep = ""),
    y = "residual",
    title = paste("Residual vs. Predicted ", transformed_response, sep = "")
   ) +
   theme(
    plot.title = element_text(hjust = 0.5, size = 11),
    axis.text.x = element_text(angle = 0)
   )
  residual_plots[[i]] <- list(residual_plot)
  if (Box_Cox_Method_should_be_performed) {
      result_of_Box_Cox_Method <- perform_Box_Cox_Method(linear_model, whether_to_plot = FALSE)
      maximum_likelihood_estimate_of_parameter_lambda <- result_of_Box_Cox_Method$maximum_likelihood_estimate_of_parameter_lambda
      lambdas[[i]] <- maximum_likelihood_estimate_of_parameter_lambda
  }
 }
 summary_of_all_MLR_models <- list(data_frame = data_frame_of_all_MLR_models, residual_plots = residual_plots, lambdas = lambdas)
 return(summary_of_all_MLR_models)
}
