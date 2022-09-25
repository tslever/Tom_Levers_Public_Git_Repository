#' @title perform_Box_Cox_Method
#' @description Performs the Box-Cox Method
#' @param linear_model A linear model
#' @param vector_of_values_of_lambda Defaults to seq(-2, 2, 0.1)
#' @param whether_to_plot Defaults to true
#' @return result_of_Box_Cox_Method
#' @examples result_of_Box_Cox_Method <- perform_Box_Cox_Method(linear_model)

#' @export
perform_Box_Cox_Method <- function(linear_model, vector_of_values_of_lambda = seq(-2, 2, 0.1), whether_to_plot = TRUE) {
    Box_Cox_plot_data <- boxcox(linear_model, lambda = vector_of_values_of_lambda, plotit = whether_to_plot, interp = TRUE, eps = 0.02, xlab = "lambda", ylab = "log-Likelihood")

    likelihoods <- Box_Cox_plot_data$y
    maximum_likelihood <- max(likelihoods)
    index_of_maximum_likelihood_in_likelihoods <- match(maximum_likelihood, likelihoods)
    parameters <- Box_Cox_plot_data$x
    the_maximum_likelihood_estimate_of_parameter_lambda <- parameters[index_of_maximum_likelihood_in_likelihoods]

    fitted_values <- linear_model$fitted.values
    logarithmicized_fitted_values <- log(fitted_values)
    sum_of_logarithmicized_fitted_values <- sum(logarithmicized_fitted_values)
    average_of_logarithmicized_fitted_values <- sum_of_logarithmicized_fitted_values / length(logarithmicized_fitted_values)
    log_of_average_of_logarithmicized_fitted_values <- log(average_of_logarithmicized_fitted_values)
    reciprocal_of_log_of_average_of_logarithmicized_fitted_values <- 1 / log_of_average_of_logarithmicized_fitted_values

    the_transformed_fitted_values <- numeric(length(fitted_values))
    if (the_maximum_likelihood_estimate_of_parameter_lambda == 0) {
        the_transformed_fitted_values <- reciprocal_of_log_of_average_of_logarithmicized_fitted_values * log(fitted_values)
    } else {
        the_transformed_fitted_values <- (fitted_values^the_maximum_likelihood_estimate_of_parameter_lambda - 1) / (the_maximum_likelihood_estimate_of_parameter_lambda * reciprocal_of_log_of_average_of_logarithmicized_fitted_values^(the_maximum_likelihood_estimate_of_parameter_lambda - 1))
    }

    result_of_Box_Cox_Method <- list(
        maximum_likelihood_estimate_of_parameter_lambda = the_maximum_likelihood_estimate_of_parameter_lambda,
        transformed_fitted_values = the_transformed_fitted_values
    )
    class(result_of_Box_Cox_Method) <- "result_of_Box_Cox_Method"
    return(result_of_Box_Cox_Method)
}
