#' @title perform_Box_Cox_Method
#' @description Performs the Box-Cox Method
#' @param linear_model A linear model
#' @param vector_of_values_of_lambda Defaults to seq(-2, 2, 0.1)
#' @param whether_to_plot Defaults to true
#' @return result_of_Box_Cox_Method
#' @examples result_of_Box_Cox_Method <- perform_Box_Cox_Method(linear_model)
#' @import MASS

#' @export
perform_Box_Cox_Method <- function(linear_model, vector_of_values_of_lambda = seq(-2, 2, 0.1), whether_to_plot = TRUE) {
    Box_Cox_plot_data <- boxcox.default(linear_model, plotit = whether_to_plot)
    likelihoods <- Box_Cox_plot_data$y
    maximum_likelihood <- max(likelihoods)
    index_of_maximum_likelihood_in_likelihoods <- match(maximum_likelihood, likelihoods)
    if (index_of_maximum_likelihood_in_likelihoods == 1 || index_of_maximum_likelihood_in_likelihoods == length(likelihoods)) {
     stop("The maximum likelihood estimate of parameter lambda is at the beginning or end of the specified range")
    }
    parameters <- Box_Cox_plot_data$x
    maximum_likelihood_estimate_of_parameter_lambda <- parameters[index_of_maximum_likelihood_in_likelihoods]
    #response_values <- linear_model$model[,1]
    #the_transformed_response_values <- Box_Cox_equation(response_values, maximum_likelihood_estimate_of_parameter_lambda)
    result_of_Box_Cox_Method <- list(
        maximum_likelihood_estimate_of_parameter_lambda = maximum_likelihood_estimate_of_parameter_lambda
        #maximum_likelihood_estimate_of_parameter_lambda = maximum_likelihood_estimate_of_parameter_lambda,
        #transformed_response_values = the_transformed_response_values
    )
    class(result_of_Box_Cox_Method) <- "result_of_Box_Cox_Method"
    return(result_of_Box_Cox_Method)
}

Box_Cox_equation <- function(values, maximum_likelihood_estimate_of_parameter_lambda) {
    logarithmicized_values <- log(values)
    sum_of_logarithmicized_values <- sum(logarithmicized_values)
    average_of_logarithmicized_values <- sum_of_logarithmicized_values / length(logarithmicized_values)
    log_of_average_of_logarithmicized_values <- log(average_of_logarithmicized_values)
    reciprocal_of_log_of_average_of_logarithmicized_values <- 1 / log_of_average_of_logarithmicized_values

    transformed_values <- numeric(length(values))
    if (maximum_likelihood_estimate_of_parameter_lambda == 0) {
        transformed_values <- reciprocal_of_log_of_average_of_logarithmicized_values * log(values)
    } else {
        transformed_values <- (values^maximum_likelihood_estimate_of_parameter_lambda - 1) / (maximum_likelihood_estimate_of_parameter_lambda * reciprocal_of_log_of_average_of_logarithmicized_values^(maximum_likelihood_estimate_of_parameter_lambda - 1))
    }
    return(transformed_values)
}

boxcox.default <- function(linear_model, lambda = seq(-2, 2, 1/10), plotit = TRUE, interp = ((m < 100)), eps = 1/50, xlab = expression(lambda), ylab = "log-Likelihood", ...) {
#boxcox.default <- function(linear_model, lambda = seq(-2, 2, 1/10), plotit = TRUE, interp = (plotit && (m < 100)), eps = 1/50, xlab = expression(lambda), ylab = "log-Likelihood", ...) {
  y <- linear_model$model[, 1]
  xqr <- linear_model$qr
  if(any(y <= 0)) {
      stop("response variable must be positive")
  }
  n <- length(y)
  ## scale y[]  {for accuracy in  y^la - 1 }:
  y <- y / exp(mean(log(y)))
  logy <- log(y) # now  ydot = exp(mean(log(y))) == 1
  xl <- loglik <- as.vector(lambda)
  m <- length(xl)
  for(i in 1L:m) {
   if(abs(la <- xl[i]) > eps)
    yt <- (y^la - 1)/la
   else yt <- logy * (1 + (la * logy)/2 *
                       (1 + (la * logy)/3 * (1 + (la * logy)/4)))
   loglik[i] <- - n/2 * log(sum(qr.resid(xqr, yt)^2))
  }
  if(interp) {
   sp <- spline(xl, loglik, n = 100)
   xl <- sp$x
   loglik <- sp$y
   m <- length(xl)
  }
  if(plotit) {
   mx <- (1L:m)[loglik == max(loglik)][1L]
   Lmax <- loglik[mx]
   lim <- Lmax - qchisq(19/20, 1)/2
   dev.hold(); on.exit(dev.flush())
   plot(xl, loglik, xlab = xlab, ylab = ylab, type = "l", ylim = range(loglik, lim))
   plims <- par("usr")
   abline(h = lim, lty = 3)
   y0 <- plims[3L]
   scal <- (1/10 * (plims[4L] - y0))/par("pin")[2L]
   scx <- (1/10 * (plims[2L] - plims[1L]))/par("pin")[1L]
   text(xl[1L] + scx, lim + scal, " 95%", xpd = TRUE)
   la <- xl[mx]
   if(mx > 1 && mx < m)
    segments(la, y0, la, Lmax, lty = 3)
   ind <- range((1L:m)[loglik > lim])
   if(loglik[1L] < lim) {
    i <- ind[1L]
    x <- xl[i - 1] + ((lim - loglik[i - 1]) *
                       (xl[i] - xl[i - 1]))/(loglik[i] - loglik[i - 1])
    segments(x, y0, x, lim, lty = 3)
   }
   if(loglik[m] < lim) {
    i <- ind[2L] + 1
    x <- xl[i - 1] + ((lim - loglik[i - 1]) *
                       (xl[i] - xl[i - 1]))/(loglik[i] - loglik[i - 1])
    segments(x, y0, x, lim, lty = 3)
   }
  }
  list(x = xl, y = loglik)
 }
