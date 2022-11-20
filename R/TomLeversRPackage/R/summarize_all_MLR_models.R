#' @title summarize_all_MLR_models
#' @description Summarizes all MLR models
#' @return The summary of all MLR models
#' @examples summary_of_all_MLR_models <- summarize_all_MLR_models()
#' @import leaps

#' @export
summarize_all_MLR_models <- function(data_set, response) {

 summary_of_all_MLR_models <- data.frame(
  number_of_predictors_in_model = integer(),
  number_of_variables_in_model = integer(),
  names_of_predictors_in_model_as_string = character(),
  residual_sum_of_squares = double(),
  r_squared = double(),
  adjusted_r_squared = double(),
  residual_mean_square = double(),
  mallows_Cp = double(),
  schwartz_bayesian_information_criterion = double(),
  stringsAsFactors = FALSE
 )
 formula <- reformulate(termlabels = ".", response = response)
 subset_selection_object <- regsubsets(formula, data = data_set, nbest = 2, really.big = TRUE)
 summary_for_subset_selection_object <- summary(subset_selection_object)
 mask_of_predictors_for_all_models <- summary_for_subset_selection_object$which
 names_of_intercept_and_predictors <- colnames(mask_of_predictors_for_all_models)
 names_of_predictors <- names_of_intercept_and_predictors[2:length(names_of_intercept_and_predictors)]
 summary_of_all_MLR_models["B_0"] <- double()
 for (name_of_predictor in names_of_predictors) {
     name_of_coefficient <- paste("B_", name_of_predictor, sep = "")
     summary_of_all_MLR_models[name_of_coefficient] <- double()
 }
 for (i in 1:nrow(mask_of_predictors_for_all_models)) {
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
  names_of_predictors_in_model_as_string <- paste(names_of_predictors_in_model, collapse = ", ")
  residual_sum_of_squares <- (summary_for_subset_selection_object$rss)[i]
  r_squared <- (summary_for_subset_selection_object$rsq)[i]
  adjusted_r_squared <- (summary_for_subset_selection_object$adjr2)[i]
  names_of_predictors_for_lm <- character()
  names_of_variables <- colnames(data_set)
  for (name_of_predictor in names_of_predictors_in_model) {
      for (name_of_variable in names_of_variables) {
          name_of_variable_is_in_name_of_predictor <- grepl(name_of_variable, name_of_predictor, fixed=TRUE)
          variable_is_not_in_names_of_predictors_for_lm <- !(name_of_variable %in% names_of_predictors_for_lm)
          if (name_of_variable_is_in_name_of_predictor && variable_is_not_in_names_of_predictors_for_lm) {
              names_of_predictors_for_lm <- append(names_of_predictors_for_lm, name_of_variable)
          }
      }
  }
  formula <- reformulate(termlabels = names_of_predictors_for_lm, response = response)
  linear_model <- lm(formula, data = data_set)
  residual_mean_square <- calculate_residual_mean_square(linear_model)
  mallows_Cp <- (summary_for_subset_selection_object$cp)[i]
  schwartz_bayesian_information_criterion <- (summary_for_subset_selection_object$bic)[i]
  linear_model_coefficients <- linear_model$coefficients
  names_of_linear_model_coefficients <- names(linear_model_coefficients)
  number_of_variables <- length(names_of_intercept_and_predictors)
  coefficients <- double(number_of_variables)
  for (name_of_linear_model_coefficient in names_of_linear_model_coefficients) {
      index_of_name_of_linear_model_coefficient_in_names_of_intercept_and_predictors <- match(name_of_linear_model_coefficient, names_of_intercept_and_predictors)
      coefficients[index_of_name_of_linear_model_coefficient_in_names_of_intercept_and_predictors] = linear_model_coefficients[name_of_linear_model_coefficient]
  }
  summary_for_model <- c(number_of_predictors_in_model, number_of_variables_in_model, names_of_predictors_in_model_as_string, residual_sum_of_squares, r_squared, adjusted_r_squared, residual_mean_square, mallows_Cp, schwartz_bayesian_information_criterion, coefficients)
  summary_of_all_MLR_models[nrow(summary_of_all_MLR_models) + 1, ] <- summary_for_model
 }
 return(summary_of_all_MLR_models)

}
