#' @export
plot_distribution <- function(data_frame, variable, color) {
 ggplot() +
  geom_histogram(
   data = data_frame,
   mapping = aes(
    x = data_frame[, variable],
    y = after_stat(density)
   ),
   alpha = 0.2,
   binwidth = density(data_frame[, variable])$bw,
   fill = color
  ) +
  geom_density(
   data = data_frame,
   mapping = aes(x = data_frame[, variable]),
   color = color
  ) +
  labs(
   x = variable,
   y = "probability density",
   title = paste("Distribution of ", variable, sep = "")
  ) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 11),
   axis.text.x = element_text(angle = 0)
  )
}
