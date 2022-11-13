library(ggplot2)
data_for_exoplanets_with_density_and_semimajor_axis <- read.csv('C:\\Users\\Tom\\Documents\\Tom_Levers_Git_Repository\\UVA\\2--Linear_Models_For_Data_Science\\Project_2\\7--Curated_Data_With_Density_And_Semimajor_Axis.csv', header = TRUE)
plot <- ggplot(
 data_for_exoplanets_with_density_and_semimajor_axis,
 aes(x = log(orbital_semimajor_axis_in_AU), y = log(density_in_grams_per_cubic_centimeter), color = type)
) +
 geom_point(alpha = 0.5) +
 labs(
  x = "orbital semimajor axis (AU)",
  y = "density (grams per cubic centimeter)",
  title = "Density vs. Orbital Semimajor Axis"
 ) +
 theme(
  plot.title = element_text(hjust = 0.5),
  axis.text.x = element_text(angle = 0)
 )
print(plot)