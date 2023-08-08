#' @title summarize_performance_of_cross_validated_models_using_dplyr
#' @description Summarizes performance of cross validated models using dplyr
#' @param type_of_model The type of model for which to summarize performance of cross validated models using dplyr
#' @param formula The formula of the model for which to summarize performance of cross validated models using dplyr
#' @param data_frame The data frame for which to summarize performance of trained and cross validated models using dplyr
#' @param optimal_lambda The optimal value of lambda for which to summarize performance of trained and cross validated logistic ridge regression models
#' @return summary_of_performance_of_cross_validated_models_using_dplyr The summary of performance of cross-validated models using dplyr
#' @examples summary_of_performance_of_cross_validated_models_using_dplyr <- summarize_performance_of_cross_validated_models_using_dplyr("Logistic Regression", Indicator ~ Red + Green + Blue, data_frame = data_frame_of_indicators_and_pixels, optimal_lambda = 0.001)
#' @import dplyr
#' @import ggplot2
#' @import rsample

#' @export
summarize_performance_of_cross_validated_models_using_dplyr <- function(type_of_model, formula, data_frame, optimal_lambda = NULL) {
 print(paste("Summary for model of type ", type_of_model, sep = ""))
 names_of_variables <- all.vars(formula)
 name_of_response <- names_of_variables[1]
 vector_of_names_of_predictors <- names_of_variables[-1]
 print(paste(names_of_variables[1], " ~ ", sep = ""))
 print(paste("    ", names_of_variables[2], sep = ""))
 number_of_variables <- length(names_of_variables)
 number_of_predictors <- length(vector_of_names_of_predictors)
 if (number_of_variables > 2) {
  for (i in 3:number_of_variables) {
   print(paste("    + ", names_of_variables[i], sep = ""))
  }
 }
 vector_of_random_indices <- sample(1:floor(nrow(data_frame)/10))
 sample_of_data_frame <- data_frame[vector_of_random_indices,]
 if (type_of_model == "Logistic Ridge Regression") {
  if (is.null(optimal_lambda)) {
   matrix_of_predictors <- as.matrix(data_frame[, vector_of_names_of_predictors])
   vector_of_response_values <- as.numeric(data_frame[, name_of_response])
   the_glmnet <- glmnet::glmnet(x = matrix_of_predictors, y = vector_of_response_values, family = "binomial", alpha = 0)
   sequence_of_lambda_values <- the_glmnet$lambda
   print(paste("sequence of lambda values = {", sequence_of_lambda_values[1], ", ", sequence_of_lambda_values[2], ", ..., ", sequence_of_lambda_values[length(sequence_of_lambda_values)], "}", sep = ""))
   the_trainControl <- caret::trainControl(method  = "cv", summaryFunction = calculate_F1_measure, allowParallel = TRUE)
   list_of_training_information <- caret::train(
    form = formula,
    data = data_frame,
    method = "glmnet",
    metric = "F1_measure",
    trControl = the_trainControl,
    tuneGrid = expand.grid(alpha = 0, lambda = sequence_of_lambda_values)
   )
   optimal_lambda = list_of_training_information$bestTune$lambda
  }
  print(paste("optimal value of lambda = ", optimal_lambda, sep = ""))
 } else if (type_of_model == "KNN") {
  the_trainControl <- caret::trainControl(method  = "cv", summaryFunction = calculate_F1_measure, allowParallel = TRUE)
  list_of_training_information <- caret::train(
   form = formula,
   data = data_frame,
   method = "knn",
   metric = "F1_measure",
   trControl = the_trainControl,
   tuneGrid = expand.grid(k = seq(from = 1, to = 25, by = 1))
  )
  print(plot(list_of_training_information))
  optimal_K = list_of_training_information$bestTune
  print(paste("optimal value of K = ", optimal_K, sep = ""))
 } else if (type_of_model == "Random Forest") {
  training_and_testing_data <- split_data_set_into_training_and_testing_data(data_frame = training_data_frame_of_indicators_and_pixels[, names_of_variables], proportion_of_training_data = 0.9)
  training_data <- training_and_testing_data$training_data
  testing_data <- training_and_testing_data$testing_data
  index_of_column_Indicator <- get_index_of_column_of_data_frame(data_frame = training_data, column_name = "Indicator")
  data_frame_of_training_predictors <- training_data[, -index_of_column_Indicator]
  data_frame_of_training_response_values <- training_data[, index_of_column_Indicator]
  data_frame_of_testing_predictors <- testing_data[, -index_of_column_Indicator]
  data_frame_of_testing_response_values <- testing_data[, index_of_column_Indicator]
  maximum_number_of_trees = 1000
  maximum_test_error_rate <- -1
  minimum_test_error_rate <- 2
  optimal_number_of_trees <- 0
  optimal_mtry <- 0
  vector_of_numbers_of_trees <- numeric(0)
  vector_of_test_error_rates <- numeric(0)
  vector_of_values_of_mtry <- factor()
  for (mtry in 1:number_of_predictors) {
   the_randomForest <- randomForest::randomForest(
    x = data_frame_of_training_predictors,
    y = data_frame_of_training_response_values,
    xtest = data_frame_of_testing_predictors,
    ytest = data_frame_of_testing_response_values,
    mtry = mtry,
    ntree = maximum_number_of_trees
   )
   data_frame_of_error_rates <- the_randomForest$test$err.rate
   vector_of_test_error_rates_for_present_mtry <- data_frame_of_error_rates[, 1]
   vector_of_test_error_rates <- c(vector_of_test_error_rates, vector_of_test_error_rates_for_present_mtry)
   vector_of_values_of_mtry <- c(vector_of_values_of_mtry, factor(rep(x = mtry, times = maximum_number_of_trees)))
   vector_of_numbers_of_trees <- c(vector_of_numbers_of_trees, 1:maximum_number_of_trees)
   minimum_test_error_rate_for_present_mtry <- min(vector_of_test_error_rates_for_present_mtry)
   if (minimum_test_error_rate_for_present_mtry < minimum_test_error_rate) {
    minimum_test_error_rate <- minimum_test_error_rate_for_present_mtry
    optimal_mtry <- mtry
    optimal_number_of_trees <- which.min(vector_of_test_error_rates_for_present_mtry)
   }
   maximum_test_error_rate_for_present_mtry <- max(vector_of_test_error_rates_for_present_mtry)
   if (maximum_test_error_rate_for_present_mtry > maximum_test_error_rate) {
    maximum_test_error_rate <- maximum_test_error_rate_for_present_mtry
   }
  }
  the_ggplot <- ggplot() +
   geom_line(
    data = data.frame(
     number_of_trees = vector_of_numbers_of_trees,
     test_error_rate = vector_of_test_error_rates,
     mtry = vector_of_values_of_mtry
    ),
    mapping = aes(
     x = number_of_trees,
     y = test_error_rate,
     color = mtry
    )
   ) +
   labs(
    x = "Number Of Trees",
    y = "Test Error Rate",
    title = "Test Error Rate Vs. Number Of Trees"
   ) +
   theme(
    plot.title = element_text(hjust = 0.5, size = 11),
   )
  print(the_ggplot)
  message <- paste("Minimum test error rate: ", minimum_test_error_rate, sep = "")
  print(message)
  message <- paste("Value of mtry corresponding to minimum test error rate: ", optimal_mtry, sep = "")
  print(message)
  message <- paste("Number of trees corresponding to minimum test error rate: ", optimal_number_of_trees, sep = "")
  print(message)
 } else if (type_of_model == "Support-Vector Machine With Linear Kernel") {
  SVM_with_linear_kernel <- list(
   type = "Classification",
   library = "e1071"
  )
  SVM_with_linear_kernel$parameters <- data.frame(
   parameter = c("cost")
  )
  SVM_with_linear_kernel$grid <- function() {}
  SVM_with_linear_kernel$fit <- function(x, y, parameters, classProbs, last, lev, wts) {
   e1071::svm(x, y, cost = parameters$cost, kernel = "linear", probability = TRUE)
  }
  SVM_with_linear_kernel$predict <- function(modelFit, newdata, submodels = NULL) {
   vector_of_predicted_values <- predict(modelFit, newdata)
   return(vector_of_predicted_values)
  }
  SVM_with_linear_kernel$prob <- function(modelFit, newdata) {
   factor_of_predictions_and_predicted_probabilities <- predict(modelFit, newdata = testing_data, probability = TRUE)
   matrix_of_predicted_probabilities <- attr(x = factor_of_predictions_and_predicted_probabilities, which = "probabilities")
   return(matrix_of_predicted_probabilities)
  }
  the_trainControl <- caret::trainControl(method = "cv", summaryFunction = calculate_F1_measure, allowParallel = TRUE)
  range_of_costs <- 10^seq(from = 1, to = 3, by = 1)
  the_tuneGrid = expand.grid(.cost = range_of_costs)
  list_of_training_information <- caret::train(
   form = formula,
   data = sample_of_data_frame,
   method = SVM_with_linear_kernel,
   metric = "F1_measure",
   trControl = the_trainControl,
   tuneGrid = the_tuneGrid
  )
  print(plot(list_of_training_information))
  optimal_cost = list_of_training_information$bestTune$cost
  print(paste("optimal cost = ", optimal_cost, sep = ""))
 } else if (type_of_model == "Support-Vector Machine With Polynomial Kernel") {
  SVM_with_polynomial_kernel <- list(
   type = "Classification",
   library = "e1071"
  )
  SVM_with_polynomial_kernel$parameters <- data.frame(
   parameter = c("cost", "degree")
  )
  SVM_with_polynomial_kernel$grid <- function() {}
  SVM_with_polynomial_kernel$fit <- function(x, y, parameters, classProbs, last, lev, wts) {
   e1071::svm(x, y, cost = parameters$cost, degree = parameters$degree, kernel = "polynomial", probability = TRUE)
  }
  SVM_with_polynomial_kernel$predict <- function(modelFit, newdata, submodels = NULL) {
   vector_of_predicted_values <- predict(modelFit, newdata)
   return(vector_of_predicted_values)
  }
  SVM_with_polynomial_kernel$prob <- function(modelFit, newdata) {
   factor_of_predictions_and_predicted_probabilities <- predict(modelFit, newdata = testing_data, probability = TRUE)
   matrix_of_predicted_probabilities <- attr(x = factor_of_predictions_and_predicted_probabilities, which = "probabilities")
   return(matrix_of_predicted_probabilities)
  }
  the_trainControl <- caret::trainControl(method = "cv", summaryFunction = calculate_F1_measure, allowParallel = TRUE)
  range_of_costs <- 10^seq(from = 1, to = 3, by = 1)
  range_of_degrees <- c(1, 2, 3)
  the_tuneGrid = expand.grid(.cost = range_of_costs, .degree = range_of_degrees)
  list_of_training_information <- caret::train(
   form = formula,
   data = sample_of_data_frame,
   method = SVM_with_polynomial_kernel,
   metric = "F1_measure",
   trControl = the_trainControl,
   tuneGrid = the_tuneGrid
  )
  print(plot(list_of_training_information))
  optimal_cost = list_of_training_information$bestTune$cost
  print(paste("optimal cost = ", optimal_cost, sep = ""))
  optimal_degree = list_of_training_information$bestTune$degree
  print(paste("optimal degree = ", optimal_degree, sep = ""))
 } else if (type_of_model == "Support-Vector Machine With Radial Kernel") {
  SVM_with_radial_kernel <- list(
   type = "Classification",
   library = "e1071"
  )
  SVM_with_radial_kernel$parameters <- data.frame(
   parameter = c("cost", "gamma")
  )
  SVM_with_radial_kernel$grid <- function() {}
  SVM_with_radial_kernel$fit <- function(x, y, parameters, classProbs, last, lev, wts) {
   e1071::svm(x, y, cost = parameters$cost, gamma = parameters$gamma, kernel = "radial", probability = TRUE)
  }
  SVM_with_radial_kernel$predict <- function(modelFit, newdata, submodels = NULL) {
   vector_of_predicted_values <- predict(modelFit, newdata)
   return(vector_of_predicted_values)
  }
  SVM_with_radial_kernel$prob <- function(modelFit, newdata) {
   factor_of_predictions_and_predicted_probabilities <- predict(modelFit, newdata = testing_data, probability = TRUE)
   matrix_of_predicted_probabilities <- attr(x = factor_of_predictions_and_predicted_probabilities, which = "probabilities")
   return(matrix_of_predicted_probabilities)
  }
  the_trainControl <- caret::trainControl(method = "cv", summaryFunction = calculate_F1_measure, allowParallel = TRUE)
  range_of_costs <- 10^seq(from = 1, to = 3, by = 1)
  range_of_values_of_gamma <- 10^seq(-1, 1, by = 1)
  the_tuneGrid = expand.grid(.cost = range_of_costs, .gamma = range_of_values_of_gamma)
  list_of_training_information <- caret::train(
   form = formula,
   data = sample_of_data_frame,
   method = SVM_with_radial_kernel,
   metric = "F1_measure",
   trControl = the_trainControl,
   tuneGrid = the_tuneGrid
  )
  print(plot(list_of_training_information))
  optimal_cost = list_of_training_information$bestTune$cost
  print(paste("optimal cost = ", optimal_cost, sep = ""))
  optimal_gamma = list_of_training_information$bestTune$gamma
  print(paste("optimal value of gamma = ", optimal_gamma, sep = ""))
 }
 generate_data_frame_of_actual_indicators_and_predicted_probabilities <-
  function(train_test_split) {
   training_data <- analysis(x = train_test_split)
   testing_data <- assessment(x = train_test_split)
   if (type_of_model == "Logistic Regression") {
    logistic_regression_classifier <- glm(
     formula = formula,
     data = training_data,
     family = binomial
    )
    if (!file.exists("logistic_regression_classifier.rds")) {
     saveRDS(logistic_regression_classifier, "logistic_regression_classifier.rds")
    }
    vector_of_predicted_probabilities <- predict(
     object = logistic_regression_classifier,
     newdata = testing_data,
     type = "response"
    )
   } else if (type_of_model == "Logistic Ridge Regression") {
    training_model_matrix <- model.matrix(object = formula, data = training_data)[, -1]
    training_vector_of_indicators <- training_data$Indicator
    logistic_regression_with_lasso_model <- glmnet::glmnet(x = training_model_matrix, y = training_vector_of_indicators, alpha = 0, family = "binomial", lambda = optimal_lambda)
    testing_model_matrix <- model.matrix(object = formula, data = testing_data)[, -1]
    vector_of_predicted_probabilities <- predict(object = logistic_regression_with_lasso_model, newx = testing_model_matrix, type = "response")
   } else if (type_of_model == "LDA" | type_of_model == "QDA") {
    if (type_of_model == "LDA") {
     model <- MASS::lda(
      formula = formula,
      data = training_data
     )
    } else if (type_of_model == "QDA") {
     model <- MASS::qda(
      formula = formula,
      data = training_data
     )
    }
    prediction <- predict(model, newdata = testing_data)
    data_frame_of_predicted_probabilities <- prediction$posterior
    index_of_column_1 <- get_index_of_column_of_data_frame(data_frame_of_predicted_probabilities, 1)
    vector_of_predicted_probabilities <- data_frame_of_predicted_probabilities[, index_of_column_1]
   } else if (type_of_model == "KNN") {
    matrix_of_values_of_predictors_for_training <- as.matrix(x = training_data[, vector_of_names_of_predictors])
    matrix_of_values_of_predictors_for_testing <- as.matrix(x = testing_data[, vector_of_names_of_predictors])
    vector_of_response_values_for_training <- training_data[, name_of_response]
    the_knn3 <- caret::knn3(
     matrix_of_values_of_predictors_for_training,
     vector_of_response_values_for_training,
     k = optimal_K
    )
    data_frame_of_predicted_probabilities <- predict(
     object = the_knn3,
     matrix_of_values_of_predictors_for_testing
    )
    index_of_column_1 <- get_index_of_column_of_data_frame(data_frame_of_predicted_probabilities, 1)
    vector_of_predicted_probabilities <- data_frame_of_predicted_probabilities[, index_of_column_1]
   } else if (type_of_model == "Random Forest") {
    index_of_column_Indicator <- get_index_of_column_of_data_frame(training_data, "Indicator")
    data_frame_of_training_predictors <- training_data[, -index_of_column_Indicator]
    data_frame_of_training_response_values <- training_data[, index_of_column_Indicator]
    the_randomForest <- randomForest::randomForest(
     formula,
     training_data,
     mtry = optimal_mtry,
     ntree = optimal_number_of_trees
    )
    matrix_of_predicted_probabilities <- predict(the_randomForest, newdata = testing_data, type = "prob")
    vector_of_predicted_probabilities <- matrix_of_predicted_probabilities[, 2]
   } else if (startsWith(type_of_model, "Support-Vector Machine")) {
    SVM = NULL
    if (type_of_model == "Support-Vector Machine With Linear Kernel") {
     SVM <- e1071::svm(
      formula,
      data = training_data,
      kernel = "linear",
      cost = optimal_cost,
      probability = TRUE
     )
    } else if (type_of_model == "Support-Vector Machine With Polynomial Kernel") {
     SVM <- e1071::svm(
      formula,
      data = training_data,
      kernel = "polynomial",
      cost = optimal_cost,
      degree = optimal_degree,
      probability = TRUE
     )
    } else if (type_of_model == "Support-Vector Machine With Radial Kernel") {
     SVM <- e1071::svm(
      formula,
      data = training_data,
      kernel = "radial",
      cost = optimal_cost,
      gamma = optimal_gamma,
      probability = TRUE
     )
    } else {
     error_message <- paste("A model of type ", type_of_model, " cannot be yet generated.", sep = "")
     stop(error_message)
    }
    factor_of_predictions_and_predicted_probabilities <- predict(SVM, newdata = testing_data, probability = TRUE)
    matrix_of_predicted_probabilities <- attr(x = factor_of_predictions_and_predicted_probabilities, which = "probabilities")
    vector_predicted_probabilities <- matrix_of_predicted_probabilities[, 2]
   } else {
    error_message <- paste("The performance of models of type ", type_of_model, " cannot be yet summarized.", sep = "")
    stop(error_message)
   }
   data_frame_of_actual_indicators_and_predicted_probabilities <- data.frame(
    actual_indicator = testing_data$Indicator,
    predicted_probability = vector_of_predicted_probabilities
   )
   colnames(data_frame_of_actual_indicators_and_predicted_probabilities) <- c("actual_indicator", "predicted_probability")
   return(data_frame_of_actual_indicators_and_predicted_probabilities)
  }
 data_frame_of_performance_metrics <-
  rsample::vfold_cv(data_frame, v = 10, repeats = 1) %>%
  mutate(
   predicted_probability = purrr::map(
    splits,
    generate_data_frame_of_actual_indicators_and_predicted_probabilities
   )
  ) %>%
  tidyr::unnest(predicted_probability) %>%
  group_by(id) %>%
  reframe(
   accuracy = provide_performance_metrics(actual_indicator, predicted_probability, "accuracy"),
   decimal_of_true_positives = provide_performance_metrics(actual_indicator, predicted_probability, "decimal of true positives"),
   F1_measure = provide_performance_metrics(actual_indicator, predicted_probability, "F1 measure"),
   FPR = provide_performance_metrics(actual_indicator, predicted_probability, "FPR"),
   number_of_negatives = provide_performance_metrics(actual_indicator, predicted_probability, "number of negatives"),
   number_of_observations = 1:length(accuracy),
   number_of_positives = provide_performance_metrics(actual_indicator, predicted_probability, "number of positives"),
   PPV = provide_performance_metrics(actual_indicator, predicted_probability, "PPV"),
   threshold = provide_performance_metrics(actual_indicator, predicted_probability, "threshold"),
   TPR = provide_performance_metrics(actual_indicator, predicted_probability, "TPR"),
  )
 data_frame_of_fold_ids_and_maximum_F1_measures <- data_frame_of_performance_metrics[complete.cases(data_frame_of_performance_metrics), ] %>% group_by(id) %>% summarise(maximum_F1_measure = max(F1_measure))
 data_frame_of_average_performance_metrics <-
  data_frame_of_performance_metrics %>%
  ungroup %>%
  group_by(number_of_observations) %>%
  reframe(
   accuracy = mean(accuracy),
   decimal_of_true_positives = mean(decimal_of_true_positives),
   F1_measure = mean(F1_measure),
   FPR = mean(FPR),
   id = "Average",
   number_of_negatives = mean(number_of_negatives),
   number_of_positives = mean(number_of_positives),
   PPV = mean(PPV),
   threshold = mean(threshold),
   TPR = mean(TPR),
  )
 data_frame_of_average_performance_metrics <- data_frame_of_average_performance_metrics[complete.cases(data_frame_of_average_performance_metrics), ]
 plot_of_performance_metrics_vs_threshold <- ggplot(
  data = data_frame_of_average_performance_metrics,
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
 data_frame_of_PPV_and_TPR <- data_frame_of_average_performance_metrics[, c("PPV", "TPR")]
 number_of_thresholds <- nrow(data_frame_of_PPV_and_TPR)
 number_of_negatives <- data_frame_of_average_performance_metrics[1, "number_of_negatives"]
 number_of_positives <- data_frame_of_average_performance_metrics[1, "number_of_positives"]
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
 data_frame_of_TPR_and_FPR <- data_frame_of_average_performance_metrics[, c("TPR", "FPR")]
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
 maximum_average_F1_measure <- max(data_frame_of_average_performance_metrics$F1_measure, na.rm = TRUE)
 index_of_column_F1_measure <- get_index_of_column_of_data_frame(data_frame_of_average_performance_metrics, "F1_measure")
 index_of_maximum_average_F1_measure <- which(data_frame_of_average_performance_metrics[, index_of_column_F1_measure] == maximum_average_F1_measure)
 data_frame_corresponding_to_maximum_average_F1_measure <-
  data_frame_of_average_performance_metrics[index_of_maximum_average_F1_measure, c("threshold", "accuracy", "TPR", "FPR", "PPV", "F1_measure")] %>%
  rename(corresponding_threshold = threshold, corresponding_accuracy = accuracy, corresponding_TPR = TPR, corresponding_FPR = FPR, corresponding_PPV = PPV, optimal_F1_measure = F1_measure)
 data_frame_corresponding_to_maximum_average_F1_measure <- data_frame_corresponding_to_maximum_average_F1_measure[1, ]
 optimal_F1_measure <- data_frame_corresponding_to_maximum_average_F1_measure$optimal_F1_measure
 #number_of_thresholds <- nrow(data_frame_of_average_performance_metrics)
 #significance_level <- 0.05
 #critical_value_tc <- qt(p = significance_level / 2, df = number_of_thresholds - 1, lower.tail = FALSE)
 #standard_error_of_optimal_F1_measure <- sqrt(optimal_F1_measure * (1 - optimal_F1_measure) / number_of_thresholds)
 #lower_bound_of_CI_for_optimal_F1_measure <- optimal_F1_measure - critical_value_tc * standard_error_of_optimal_F1_measure
 #upper_bound_of_CI_for_optimal_F1_measure <- optimal_F1_measure + critical_value_tc * standard_error_of_optimal_F1_measure
 data_frame_of_optimal_performance_metrics <- data.frame(
  corresponding_threshold = data_frame_corresponding_to_maximum_average_F1_measure$corresponding_threshold,
  alpha = ifelse(test = type_of_model == "Logistic Ridge Regression", yes = 0, no = NA),
  optimal_lambda = ifelse(test = type_of_model == "Logistic Ridge Regression", yes = optimal_lambda, no = NA),
  optimal_K = ifelse(test = type_of_model == "KNN", yes = optimal_K, no = NA),
  optimal_PR_AUC = MESS::auc(data_frame_of_PPV_and_TPR$TPR, data_frame_of_PPV_and_TPR$PPV),
  optimal_ROC_AUC = MESS::auc(data_frame_of_TPR_and_FPR$FPR, data_frame_of_TPR_and_FPR$TPR),
  corresponding_accuracy = data_frame_corresponding_to_maximum_average_F1_measure$corresponding_accuracy,
  corresponding_TPR = data_frame_corresponding_to_maximum_average_F1_measure$corresponding_TPR,
  corresponding_FPR = data_frame_corresponding_to_maximum_average_F1_measure$corresponding_FPR,
  corresponding_PPV = data_frame_corresponding_to_maximum_average_F1_measure$corresponding_PPV,
  optimal_F1_measure = optimal_F1_measure
 )
 summary_of_performance_of_cross_validated_models <- list(
  plot_of_performance_metrics_vs_threshold = plot_of_performance_metrics_vs_threshold,
  PR_curve = PR_curve,
  ROC_curve = ROC_curve,
  data_frame_of_optimal_performance_metrics = data_frame_of_optimal_performance_metrics,
  data_frame_of_fold_ids_and_maximum_F1_measures = data_frame_of_fold_ids_and_maximum_F1_measures
 )
 return(summary_of_performance_of_cross_validated_models)
}

# lev = vector_of_factor_levels_that_correspond_to_results
# model = method_specified_in_train
calculate_F1_measure <- function(data_frame_containing_actual_and_predicted_labels, lev = NULL, model = NULL) {
 F1_measure <- MLmetrics::F1_Score(y_true = data_frame_containing_actual_and_predicted_labels$obs, y_pred = data_frame_containing_actual_and_predicted_labels$pred, positive = lev[1])
 vector_with_F1_measure <- c(F1_measure = F1_measure)
 return(vector_with_F1_measure)
}
