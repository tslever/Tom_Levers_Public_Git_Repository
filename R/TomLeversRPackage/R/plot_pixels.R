#' @title plot_pixels
#' @description Plots pixels
#' @return the_ggplot The ggplot
#' @examples the_ggplot <- plot_pixels(data_frame)
#' @import ggplot2

#' @export
plot_pixels <- function(data_frame) {
 data_frame$r <- scale_to_between_0_and_1(data_frame$r)
 data_frame$g <- scale_to_between_0_and_1(data_frame$g)
 data_frame$b <- scale_to_between_0_and_1(data_frame$b)
 maximum_x <- max(data_frame$x)
 maximum_y <- max(data_frame$y)
 maximum_ordinate <- max(maximum_x, maximum_y)
 the_ggplot <- ggplot(
  data = data_frame,
  mapping = aes(
   x = x,
   y = y,
   col = rgb(r, g, b)
  )
 ) +
  geom_point() +
  scale_color_identity() +
  xlim(0, maximum_ordinate) +
  ylim(0, maximum_ordinate)
 return(the_ggplot)
}
