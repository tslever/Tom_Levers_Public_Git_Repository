#' @export
calculate_probability_for_randomization_test <- function(
  data_frame_of_fold_ids_and_maximum_F1_measures_for_Logistic_Regression,
  data_frame_of_fold_ids_and_maximum_F1_measures_for_KNN
) {
 data_frame_of_fold_ids_and_maximum_F1_measures <- data.frame(
  fold_id = data_frame_of_fold_ids_and_maximum_F1_measures_for_Logistic_Regression$id,
  maximum_F1_measure_for_Logistic_Regression =
   data_frame_of_fold_ids_and_maximum_F1_measures_for_Logistic_Regression$
   maximum_F1_measure,
  maximum_F1_measure_for_KNN =
   data_frame_of_fold_ids_and_maximum_F1_measures_for_KNN$maximum_F1_measure
 )
 data_frame_of_fold_ids_and_maximum_F1_measures

 average_maximum_F1_measure_for_Logistic_Regression <- mean(
  data_frame_of_fold_ids_and_maximum_F1_measures$
   maximum_F1_measure_for_Logistic_Regression
 )
 average_maximum_F1_measure_for_KNN <-
  mean(data_frame_of_fold_ids_and_maximum_F1_measures$maximum_F1_measure_for_KNN)
 actual_difference_between_averages <-
  average_maximum_F1_measure_for_Logistic_Regression -
  average_maximum_F1_measure_for_KNN
 count <- 0
 number_of_iterations <- 100
 number_of_folds <- nrow(data_frame_of_fold_ids_and_maximum_F1_measures)
 for (i in 1:number_of_iterations) {
  data_frame_with_swapped_maximum_F1_measures <-
   data.frame(data_frame_of_fold_ids_and_maximum_F1_measures)
  for (j in 1:number_of_folds) {
   random_binary_digit <-
    sample(x = 0:1, size = 1, replace = TRUE, prob = c(0.5, 0.5))
   if (random_binary_digit == 1) {
    archived_value <- data_frame_with_swapped_maximum_F1_measures[
     j,
     "maximum_F1_measure_for_KNN"
    ]
    data_frame_with_swapped_maximum_F1_measures[
     j,
     "maximum_F1_measure_for_KNN"
    ] <-
     data_frame_with_swapped_maximum_F1_measures[
      j,
      "maximum_F1_measure_for_Logistic_Regression"
     ]
    data_frame_with_swapped_maximum_F1_measures[
     j,
     "maximum_F1_measure_for_Logistic_Regression"
    ] <- archived_value
   }
  }
  average_maximum_F1_measure_for_Logistic_Regression <- mean(
   data_frame_with_swapped_maximum_F1_measures$
    maximum_F1_measure_for_Logistic_Regression
  )
  average_maximum_F1_measure_for_KNN <- mean(
   data_frame_with_swapped_maximum_F1_measures$maximum_F1_measure_for_KNN
  )
  modified_difference_between_averages <-
   average_maximum_F1_measure_for_Logistic_Regression -
   average_maximum_F1_measure_for_KNN
  modified_difference_between_averages
  if (modified_difference_between_averages > actual_difference_between_averages) {
   count <- count + 1
  }
 }
 probability_that_modified_difference_is_greater_than_actual_difference <-
  count / number_of_iterations
 return(probability_that_modified_difference_is_greater_than_actual_difference)
}
