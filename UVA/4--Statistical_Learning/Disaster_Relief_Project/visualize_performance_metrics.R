training_data_frame_of_classes_and_pixels <- read.csv(
 file = "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Training_Data_Frame_Of_Classes_And_Pixels.csv"
)
holdout_data_frame_of_classes_and_pixels <- read.csv(
 file = "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/Holdout_Data_Frame_Of_Classes_And_Pixels.csv"
)
training_data_frame_of_indicators_and_pixels <- NULL
holdout_data_frame_of_indicators_and_pixels <- NULL
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
 write.csv(holdout_data_frame_of_indicators_and_pixels, "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/Holdout_Data_Frame_Of_Indicators_And_Pixels.csv", row.names = FALSE)
} else {
 training_data_frame_of_indicators_and_pixels <- read.csv("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Training_Data_Frame_Of_Indicators_And_Pixels.csv", header = TRUE)
 training_data_frame_of_indicators_and_pixels$Indicator <- factor(training_data_frame_of_indicators_and_pixels$Indicator)
 holdout_data_frame_of_indicators_and_pixels <- read.csv("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/Holdout_Data_Frame_Of_Indicators_And_Pixels.csv", header = TRUE)
 holdout_data_frame_of_indicators_and_pixels$Indicator <- factor(holdout_data_frame_of_indicators_and_pixels$Indicator)
}

number_of_training_observations <- nrow(training_data_frame_of_indicators_and_pixels)
number_of_holdout_observations <- nrow(holdout_data_frame_of_indicators_and_pixels)
vector_of_random_indices <- sample(x = 1:number_of_holdout_observations, size = number_of_training_observations)
slice_of_holdout_data_frame <- holdout_data_frame_of_indicators_and_pixels[vector_of_random_indices, ]

formula = Indicator ~ Normalized_Square_Of_Blue + Normalized_Square_Of_Red + Normalized_Square_Root_Of_Blue
type_of_model <- "Logistic Regression"
list_of_visualizations_of_performance_metrics <- TomLeversRPackage::visualize_performance_metrics(formula, slice_of_holdout_data_frame, training_data_frame_of_indicators_and_pixels, "Logistic Regression", list_of_hyperparameters = NULL)
print(list_of_visualizations_of_performance_metrics$PR_curve)
print(list_of_visualizations_of_performance_metrics$ROC_curve)
print(list_of_visualizations_of_performance_metrics$plot_of_performance_metrics_vs_threshold)
print(list_of_visualizations_of_performance_metrics$data_frame_of_optimal_performance_metrics)