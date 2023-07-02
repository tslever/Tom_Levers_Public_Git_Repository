test_that("summarize_performance_of_cross_validated_models_using_dplyr works",  {
 data_frame_of_classes_and_pixels <- read.csv(
  file = "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Data_Frame_Of_Classes_And_Pixels.csv"
 )
 number_of_observations <- nrow(data_frame_of_classes_and_pixels)
 column_of_indicators <- rep(0, number_of_observations)
 condition <- data_frame_of_classes_and_pixels$Class == "Blue Tarp"
 column_of_indicators[condition] <- 1
 factor_of_indicators <- factor(column_of_indicators)
 data_frame_of_indicators_and_pixels <- data.frame(
  Indicator = factor_of_indicators,
  Normalized_Red = normalize_vector(data_frame_of_classes_and_pixels[, "Red"]),
  Normalized_Green = normalize_vector(data_frame_of_classes_and_pixels[, "Green"]),
  Normalized_Blue = normalize_vector(data_frame_of_classes_and_pixels[, "Blue"]),
  Normalized_Natural_Logarithm_Of_Red =
   normalize_vector(log(data_frame_of_classes_and_pixels[, "Red"])),
  Normalized_Natural_Logarithm_Of_Green =
   normalize_vector(log(data_frame_of_classes_and_pixels[, "Green"])),
  Normalized_Natural_Logarithm_Of_Blue =
   normalize_vector(log(data_frame_of_classes_and_pixels[, "Blue"])),
  Normalized_Square_Root_Of_Red =
   normalize_vector(sqrt(data_frame_of_classes_and_pixels[, "Red"])),
  Normalized_Square_Root_Of_Green =
   normalize_vector(sqrt(data_frame_of_classes_and_pixels[, "Green"])),
  Normalized_Square_Root_Of_Blue =
   normalize_vector(sqrt(data_frame_of_classes_and_pixels[, "Blue"])),
  Normalized_Square_Of_Red =
   normalize_vector(I(data_frame_of_classes_and_pixels[, "Red"]^2)),
  Normalized_Square_Of_Green =
   normalize_vector(I(data_frame_of_classes_and_pixels[, "Green"]^2)),
  Normalized_Square_Of_Blue =
   normalize_vector(I(data_frame_of_classes_and_pixels[, "Blue"]^2)),
  Normalized_Interaction_Of_Red_And_Green = normalize_vector(
   as.numeric(
    interaction(
     data_frame_of_classes_and_pixels$Red,
     data_frame_of_classes_and_pixels$Green
    )
   )
  ),
  Normalized_Interaction_Of_Red_And_Blue = normalize_vector(
   as.numeric(
    interaction(
     data_frame_of_classes_and_pixels$Red,
     data_frame_of_classes_and_pixels$Blue
    )
   )
  ),
  Normalized_Interaction_Of_Green_And_Blue = normalize_vector(
   as.numeric(
    interaction(
     data_frame_of_classes_and_pixels$Green,
     data_frame_of_classes_and_pixels$Blue
    )
   )
  )
 )
 vector_of_random_indices <- sample(1:number_of_observations)
 data_frame_of_indicators_and_pixels <-
  data_frame_of_indicators_and_pixels[vector_of_random_indices, ]

 summary_of_performance <- summarize_performance_of_cross_validated_models_using_dplyr(
  type_of_model = "Logistic Regression",
  formula = Indicator ~ Normalized_Interaction_Of_Red_And_Blue + Normalized_Interaction_Of_Green_And_Blue + Normalized_Square_Root_Of_Blue,
  data_frame = data_frame_of_indicators_and_pixels
 )
})
