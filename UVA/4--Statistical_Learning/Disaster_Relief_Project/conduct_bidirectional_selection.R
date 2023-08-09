training_data_frame_of_classes_and_pixels <- read.csv(
file = "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Training_Data_Frame_Of_Classes_And_Pixels.csv"
)
holdout_data_frame_of_classes_and_pixels <- read.csv(
file = "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data_Frame_Of_Classes_And_Pixels.csv"
)
training_data_frame_of_indicators_and_pixels <- NULL
should_generate_training_and_holdout_data_frames_of_indicators_and_pixels <- FALSE
if (should_generate_training_and_holdout_data_frames_of_indicators_and_pixels) {
composite_data_frame_of_classes_and_pixels <- rbind(training_data_frame_of_classes_and_pixels, holdout_data_frame_of_classes_and_pixels)
number_of_training_observations <- nrow(training_data_frame_of_classes_and_pixels)
number_of_holdout_observations <- nrow(holdout_data_frame_of_classes_and_pixels)
number_of_training_and_holdout_observations <- nrow(composite_data_frame_of_classes_and_pixels)
column_of_indicators <- rep(0, number_of_training_and_holdout_observations)
condition <- composite_data_frame_of_classes_and_pixels$Class == "Blue Tarp"
column_of_indicators[condition] <- 1
factor_of_indicators <- factor(column_of_indicators)
composite_data_frame_of_indicators_and_pixels <- data.frame(
 Indicator = factor_of_indicators,
 Normalized_Red = normalize_vector(composite_data_frame_of_classes_and_pixels[, "Red"]),
 Normalized_Green = normalize_vector(composite_data_frame_of_classes_and_pixels[, "Green"]),
 Normalized_Blue = normalize_vector(composite_data_frame_of_classes_and_pixels[, "Blue"]),
 Normalized_Natural_Logarithm_Of_Red = normalize_vector(log(composite_data_frame_of_classes_and_pixels[, "Red"])),
 Normalized_Natural_Logarithm_Of_Green = normalize_vector(log(composite_data_frame_of_classes_and_pixels[, "Green"])),
 Normalized_Natural_Logarithm_Of_Blue = normalize_vector(log(composite_data_frame_of_classes_and_pixels[, "Blue"])),
 Normalized_Square_Root_Of_Red = normalize_vector(sqrt(composite_data_frame_of_classes_and_pixels[, "Red"])),
 Normalized_Square_Root_Of_Green = normalize_vector(sqrt(composite_data_frame_of_classes_and_pixels[, "Green"])),
 Normalized_Square_Root_Of_Blue = normalize_vector(sqrt(composite_data_frame_of_classes_and_pixels[, "Blue"])),
 Normalized_Square_Of_Red = normalize_vector(I(composite_data_frame_of_classes_and_pixels[, "Red"]^2)),
 Normalized_Square_Of_Green = normalize_vector(I(composite_data_frame_of_classes_and_pixels[, "Green"]^2)),
 Normalized_Square_Of_Blue = normalize_vector(I(composite_data_frame_of_classes_and_pixels[, "Blue"]^2)),
 Normalized_Interaction_Of_Red_And_Green = normalize_vector(
  as.numeric(
   interaction(
    composite_data_frame_of_classes_and_pixels$Red,
    composite_data_frame_of_classes_and_pixels$Green
   )
  )
 ),
 Normalized_Interaction_Of_Red_And_Blue = normalize_vector(
  as.numeric(
   interaction(
    composite_data_frame_of_classes_and_pixels$Red,
    composite_data_frame_of_classes_and_pixels$Blue
   )
  )
 ),
 Normalized_Interaction_Of_Green_And_Blue = normalize_vector(
  as.numeric(
   interaction(
    composite_data_frame_of_classes_and_pixels$Green,
    composite_data_frame_of_classes_and_pixels$Blue
   )
  )
 )
)
training_data_frame_of_indicators_and_pixels <- composite_data_frame_of_indicators_and_pixels[1:number_of_training_observations, ]
holdout_data_frame_of_indicators_and_pixels <- composite_data_frame_of_indicators_and_pixels[(number_of_training_observations + 1):number_of_training_and_holdout_observations,]
set.seed(1)
vector_of_random_indices <- sample(1:number_of_training_observations)
training_data_frame_of_indicators_and_pixels <- training_data_frame_of_indicators_and_pixels[vector_of_random_indices, ]
set.seed(1)
vector_of_random_indices <- sample(1:number_of_holdout_observations)
holdout_data_frame_of_indicators_and_pixels <- holdout_data_frame_of_indicators_and_pixels[vector_of_random_indices, ]
write.csv(training_data_frame_of_indicators_and_pixels, "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Training_Data_Frame_Of_Indicators_And_Pixels.csv", row.names = FALSE)
write.csv(holdout_data_frame_of_indicators_and_pixels, "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data_Frame_Of_Indicators_And_Pixels.csv", row.names = FALSE)
} else {
training_data_frame_of_indicators_and_pixels <- read.csv("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Training_Data_Frame_Of_Indicators_And_Pixels.csv", header = TRUE)
training_data_frame_of_indicators_and_pixels$Indicator <- factor(training_data_frame_of_indicators_and_pixels$Indicator)
holdout_data_frame_of_indicators_and_pixels <- read.csv("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data_Frame_Of_Indicators_And_Pixels.csv", header = TRUE)
holdout_data_frame_of_indicators_and_pixels$Indicator <- factor(holdout_data_frame_of_indicators_and_pixels$Indicator)
}

get_optimal_F1_measure_for_formula <- function(formula) {
 vector_of_names_of_variables <- all.vars(formula)
 vector_of_names_of_predictors <- vector_of_names_of_variables[-1]
 number_of_predictors <- length(vector_of_names_of_predictors)
 full_model_matrix <-
  model.matrix(object = formula, data = training_data_frame_of_indicators_and_pixels)[, -1]
 sequence_of_lambda_values <- exp(seq(-9, -8, length = 100))
 set.seed(1)
 if (number_of_predictors == 1) {
  full_model_matrix <- cbind(0, full_model_matrix)
 }
 the_cv.glmnet <- glmnet::cv.glmnet(
  x = full_model_matrix,
  y = training_data_frame_of_indicators_and_pixels$Indicator,
  family = "binomial",
  type.measure = "class",
  alpha = 0,
  lambda = sequence_of_lambda_values
 )
 data_frame <- data.frame(
  lambda = the_cv.glmnet$lambda,
  misclassification_error_rate = the_cv.glmnet$cvm,
  maximum_misclassification_error_rate = the_cv.glmnet$cvup,
  minimum_misclassification_error_rate = the_cv.glmnet$cvlo
 )
 library(ggplot2)
 the_ggplot <- ggplot(
  data = data_frame,
  mapping = aes(
   x = lambda,
   y = misclassification_error_rate,
   ymin = minimum_misclassification_error_rate,
   ymax = maximum_misclassification_error_rate
  )
 ) +
  geom_point() +
  scale_x_log10() +
  geom_errorbar() +
  labs(
   x = "lambda",
   y = "Misclassification Error Rate",
   title = "Misclassification Error Rate Vs. lambda"
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
  )
 print(the_ggplot)
 optimal_lambda <- the_cv.glmnet$lambda.min
 print(optimal_lambda)
 summary_of_performance <- TomLeversRPackage::summarize_performance_of_cross_validated_models_using_dplyr(
  type_of_model = "Logistic Ridge Regression",
  formula = formula,
  data_frame = training_data_frame_of_indicators_and_pixels,
  optimal_lambda = optimal_lambda
 )
 optimal_F1_measure_for_present_formula <-
  summary_of_performance$data_frame_of_optimal_performance_metrics$optimal_F1_measure
 return(optimal_F1_measure_for_present_formula)
}
optimal_formula_string <- NULL
optimal_F1_measure <- -1
optimal_vector_of_predictors <- NULL
vector_of_names_of_predictors <- names(training_data_frame_of_indicators_and_pixels)[-1]
for (name_1 in vector_of_names_of_predictors) {
 for (name_2 in vector_of_names_of_predictors) {
  if (name_2 == name_1) {
   print("name_1 and name_2 were the same; continuing")
   next
  }
  formula_string <- paste("Indicator ~ ", name_1, " + ", name_2, sep = "")
  formula <- as.formula(formula_string)
  optimal_F1_measure_for_present_formula <- get_optimal_F1_measure_for_formula(formula)
  print(optimal_F1_measure_for_present_formula)
  if (optimal_F1_measure_for_present_formula > optimal_F1_measure) {
   optimal_F1_measure <- optimal_F1_measure_for_present_formula
   optimal_formula_string <- formula_string
   optimal_vector_of_predictors <- c(name_1, name_2)
  }
 }
}
optimal_formula_string <- "Indicator ~ Normalized_Natural_Logarithm_Of_Blue + Normalized_Square_Root_Of_Red"
optimal_formula <- as.formula(optimal_formula_string)
optimal_F1_measure <- get_optimal_F1_measure_for_formula(optimal_formula)
print(optimal_formula_string)
print(optimal_F1_measure)
print("-----")
vector_of_names_of_predictors <- names(training_data_frame_of_indicators_and_pixels)[-1]
for (name in vector_of_names_of_predictors) {
 formula_string <- paste(optimal_formula_string, " + ", name, sep = "")
 print(formula_string)
 formula <- as.formula(formula_string)
 optimal_F1_measure_for_present_formula <- get_optimal_F1_measure_for_formula(formula)
 print(formula_string)
 print(optimal_F1_measure_for_present_formula)
 print("-----")
}