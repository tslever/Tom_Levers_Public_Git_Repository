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

get_optimal_F1_measure_for_formula <- function(type_of_model, formula) {
 out <- tryCatch(
  {
   summary_of_performance <- TomLeversRPackage::summarize_performance_of_cross_validated_classifiers(
    type_of_model = type_of_model,
    formula = formula,
    data_frame = training_data_frame_of_indicators_and_pixels
   )
   optimal_F1_measure <-
    summary_of_performance$data_frame_of_optimal_performance_metrics$optimal_F1_measure
   return(optimal_F1_measure)
  },
  error = function(cond) {
   message("Here's the original error message:")
   print(cond)
   cat(" \n")
   return("result indicating error")
  }
 )
}
optimal_vector_of_names_of_predictors <- character(0)
optimal_F1_measure <- -1
vector_of_names_of_all_variables <- names(training_data_frame_of_indicators_and_pixels)
vector_of_names_of_all_predictors <- vector_of_names_of_all_variables[-1]
optimal_F1_measure_was_adjusted <- TRUE
type_of_model <- "Random Forest"
iteration <- 1
every_optimal_F1_measure_is_result_indicating_error <- TRUE
while (optimal_F1_measure_was_adjusted) {
    print(paste("Iteration: ", iteration, sep = ""))
    optimal_F1_measure_was_adjusted <- FALSE
    list_of_potential_optimal_vectors_of_names_of_predictors = list()
    vector_of_potential_optimal_F1_measures <- numeric(0)
    for (name in vector_of_names_of_all_predictors) {
        potential_optimal_vector_of_names_of_predictors <- c(optimal_vector_of_names_of_predictors, name)
        predictor_string <- paste(potential_optimal_vector_of_names_of_predictors, collapse = " + ")
        formula_string <- paste("Indicator ~ ", predictor_string, sep = "")
        print(formula_string)
        formula <- as.formula(formula_string)
        optimal_F1_measure_for_present_predictor <- get_optimal_F1_measure_for_formula(type_of_model, formula)
        if (optimal_F1_measure_for_present_predictor != "result indicating error") {
            every_optimal_F1_measure_is_result_indicating_error <- FALSE
            print(paste("Optimal F1 measure for present predictor: ", optimal_F1_measure_for_present_predictor, sep = ""))
            if (optimal_F1_measure_for_present_predictor > optimal_F1_measure) {
                vector_of_potential_optimal_F1_measures <- append(vector_of_potential_optimal_F1_measures, optimal_F1_measure_for_present_predictor)
                list_of_potential_optimal_vectors_of_names_of_predictors <- c(list_of_potential_optimal_vectors_of_names_of_predictors, list(potential_optimal_vector_of_names_of_predictors))
            }
        }
    }
    if (length(vector_of_potential_optimal_F1_measures) > 0) {
        index_of_maximum_potential_optimal_F1_measure <- which.max(vector_of_potential_optimal_F1_measures)
        maximum_potential_optimal_F1_measure <- vector_of_potential_optimal_F1_measures[index_of_maximum_potential_optimal_F1_measure]
        optimal_F1_measure <- maximum_potential_optimal_F1_measure
        potential_optimal_vector_of_names_of_predictors_corresponding_to_maximum_potential_optimal_F1_measure <- list_of_potential_optimal_vectors_of_names_of_predictors[[index_of_maximum_potential_optimal_F1_measure]]
        optimal_vector_of_names_of_predictors <- potential_optimal_vector_of_names_of_predictors_corresponding_to_maximum_potential_optimal_F1_measure
    }
    if (iteration == 1 & every_optimal_F1_measure_is_result_indicating_error) {
        for (name_1 in vector_of_names_of_all_predictors) {
         for (name_2 in vector_of_names_of_all_predictors) {
          if (name_2 == name_1) {
           next
          }
          potential_optimal_vector_of_names_of_predictors <- c(name_1, name_2)

          predictor_string <- paste(potential_optimal_vector_of_names_of_predictors, collapse = " + ")
          formula_string <- paste("Indicator ~ ", predictor_string, sep = "")
          print(formula_string)
          formula <- as.formula(formula_string)
          optimal_F1_measure_for_present_predictor <- get_optimal_F1_measure_for_formula(type_of_model, formula)
          if (optimal_F1_measure_for_present_predictor != "result indicating error") {
           every_optimal_F1_measure_is_result_indicating_error <- FALSE
           print(paste("Optimal F1 measure for present predictor: ", optimal_F1_measure_for_present_predictor, sep = ""))
           if (optimal_F1_measure_for_present_predictor > optimal_F1_measure) {
            optimal_F1_measure <- optimal_F1_measure_for_present_predictor
            optimal_F1_measure_was_adjusted <- TRUE
            optimal_vector_of_names_of_predictors <- potential_optimal_vector_of_names_of_predictors
           }
          }
         }
        }
    }
    if (length(optimal_vector_of_names_of_predictors) > 2) {
        for (i in 1:(length(optimal_vector_of_names_of_predictors) - 2)) {
            potential_optimal_vector_of_names_of_predictors <- setdiff(optimal_vector_of_names_of_predictors, optimal_vector_of_names_of_predictors[i])

            predictor_string <- paste(potential_optimal_vector_of_names_of_predictors, collapse = " + ")
            formula_string <- paste("Indicator ~ ", predictor_string, sep = "")
            print(formula_string)
            formula <- as.formula(formula_string)
            optimal_F1_measure_for_present_predictor <- get_optimal_F1_measure_for_formula(type_of_model, formula)
            if (optimal_F1_measure_for_present_predictor == "result indicating error") {
                next
            }
            print(paste("Optimal F1 measure for present predictor: ", optimal_F1_measure_for_present_predictor, sep = ""))
            if (optimal_F1_measure_for_present_predictor > optimal_F1_measure) {
                optimal_F1_measure <- optimal_F1_measure_for_present_predictor
                optimal_F1_measure_was_adjusted <- TRUE
                optimal_vector_of_names_of_predictors <- potential_optimal_vector_of_names_of_predictors
            }
        }
    }
    iteration <- iteration + 1
}
print("Optimal vector of names of predictors:")
print(optimal_vector_of_names_of_predictors)
print(paste("Optimal F1 measure: ", optimal_F1_measure, sep = ""))
