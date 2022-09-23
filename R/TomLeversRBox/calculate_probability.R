box::use(./print.message[print.message])
box::register_S3_method('print', 'message', print.message)
box::use(stats[pnorm])

#' @export
calculate_probability <- function() {
    message <- paste("Probability: ", round(pnorm(1.644854, 0, 1, lower.tail = TRUE), 2), sep = "")
    class(message) <- "message"
    return(message)
}
