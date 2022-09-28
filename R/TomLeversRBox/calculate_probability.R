box::use(stats[pnorm])

#' @title calculate_probability
#' @description Calculates the probability of a test statistic being less than a specific quantile of a standard normal distribution
#' @export
calculate_probability <- function() {
    message <- paste("Probability: ", round(pnorm(1.644854, 0, 1, lower.tail = TRUE), 2), sep = "")
    class(message) <- "message"
    return(message)
}