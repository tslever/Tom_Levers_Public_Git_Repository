box::use(stats[pnorm])

#' @export
calculate_probability <- function() {
    message <- paste("Probability: ", round(pnorm(1.644854, 0, 1, lower.tail = TRUE), 2), sep = "")
    class(message) <- "message"
    return(message)
}