#' @title scale_to_between_0_and_1
#' @description Scales to between 0 and 1
#' @param the_vector The vector to scale to between 0 and 1
#' @return scaled_vector The scaled vector
#' @examples scaled_vector <- scale_to_between_0_and_1(the_vector)

#' @export
scale_to_between_0_and_1 <- function(the_vector) {
 scaled_vector <- (the_vector - min(the_vector)) / (max(the_vector) - min(the_vector))
 return(scaled_vector)
}
