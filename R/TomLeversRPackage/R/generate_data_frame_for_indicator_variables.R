#' @title generate_data_frame_for_indicator_variables
#' @description Generates a data frame for indicator variables based on a factor of values of a categorical variable
#' @param factor_of_values_of_categorical_variable A factor of values of a categorical variable
#' @return A data frame for indicator variables based on a factor of values of a categorical variable
#' @examples data_frame_for_indicator_variables <- generate_data_frame_for_indicator_variables(penguins$species)

#' @export
generate_data_frame_for_indicator_variables <- function(factor_of_values_of_categorical_variable) {
    data_frame_of_indicator_variables <- contrasts(factor_of_values_of_categorical_variable)
    vector_of_subscripts_of_indicator_variables <- colnames(data_frame_of_indicator_variables)
    vector_of_indicator_variables <- character(length(vector_of_subscripts_of_indicator_variables))
    for (i in 1:length(vector_of_subscripts_of_indicator_variables)) {
        vector_of_indicator_variables[i] <- paste("I_", vector_of_subscripts_of_indicator_variables[i], sep = "")
    }
    colnames(data_frame_of_indicator_variables) <- vector_of_indicator_variables
    return(data_frame_of_indicator_variables)
}
