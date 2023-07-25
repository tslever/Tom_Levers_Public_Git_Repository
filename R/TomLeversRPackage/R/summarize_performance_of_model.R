#' @export
summarize_performance_of_model <- function(vector_of_actual_values, vector_of_predicted_values) {
    confusion_matrix <- table(vector_of_actual_values, vector_of_predicted_values)
    number_of_false_negatives <- confusion_matrix[2, 1]
    number_of_false_positives <- confusion_matrix[1, 2]
    number_of_false_predictions <-
        number_of_false_negatives + number_of_false_positives
    number_of_true_negatives <- confusion_matrix[1, 1]
    number_of_true_positives <- confusion_matrix[2, 2]
    number_of_predictions <-
        number_of_false_negatives +
        number_of_false_positives +
        number_of_true_negatives +
        number_of_true_positives
    error_rate <- number_of_false_predictions / number_of_predictions
    summary_of_performance <- list(
        confusion_matrix = confusion_matrix,
        error_rate = error_rate
    )
    return(summary_of_performance)
}
