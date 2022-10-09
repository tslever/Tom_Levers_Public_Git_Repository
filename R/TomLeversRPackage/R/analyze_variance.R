#' @title analyze_variance
#' @description Analyzes the variance of a linear model
#' @param linear_model The linear model to analyze
#' @return The analysis of variance for the linear model
#' @examples analysis_of_variance <- analyze_variance(lm(iris$Sepal.Length ~ iris$Sepal.Width, data = iris))
#' @import stringr

#' @export
analyze_variance <- function(linear_model) {
    analysis <- capture.output(anova(linear_model))

    number_of_observations <- nobs(linear_model)
    total_of_degrees_of_freedom <- number_of_observations - 1
    response_values <- linear_model$model[,1]
    total_sum_of_squares <- (t(response_values) %*% response_values) - ((sum(response_values)^2) / number_of_observations)
    line_with_totals <- paste("DFT: ", total_of_degrees_of_freedom, ", SST: ", total_sum_of_squares, sep = "")
    analysis <- append(analysis, line_with_totals)

    residuals <- linear_model$residuals
    residual_sum_of_squares <- t(residuals) %*% residuals
    number_of_variables <- length(names(coefficients))
    residual_mean_square <- residual_sum_of_squares / (number_of_observations - number_of_variables)
    total_mean_square <- total_sum_of_squares / (number_of_observations - 1)
    coefficient_of_determination_R2 <- 1 - (residual_sum_of_squares / total_sum_of_squares)
    adjusted_coefficient_of_determination_R2 <- 1 - (residual_mean_square / total_mean_square)
    line_with_adjusted_coefficient_of_determination_R2 <- paste("R2: ", coefficient_of_determination_R2, ", Adjusted R2: ", adjusted_coefficient_of_determination_R2, sep = "")
    analysis <- append(analysis, line_with_adjusted_coefficient_of_determination_R2)

    line_with_number_of_observations <- paste("Number of observations: ", number_of_observations, sep = "")
    analysis <- append(analysis, line_with_number_of_observations)

    analysis <- paste(analysis, collapse = "\n")
    class(analysis) <- "analysis_of_variance"
    return(analysis)
}


#' @title print.analysis_of_variance
#' @description Adds to the generic function print the analysis_of_variance method print.analysis_of_variance
#' @param analysis The analysis to print
#' @examples print(analysis_of_variance)
#' @export

print.analysis_of_variance <- function(analysis) {
    cat(analysis)
}
