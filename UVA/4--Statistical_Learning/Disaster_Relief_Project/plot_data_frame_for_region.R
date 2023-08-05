vector_of_paths_to_data_files_for_region <- c(
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir078_ROI_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir078_ROI_NON_Blue_Tarps.txt"
)
generate_data_set_for_region <- function(vector_of_paths_to_data_files_for_region) {
 number_of_paths_to_data_files_for_region <- length(vector_of_paths_to_data_files_for_region)
 header <- c("ID", "X", "Y", "Map X", "Map Y", "Lat", "Lon", "B1", "B2", "B3")
 data_frame_for_region <- data.frame(matrix(ncol = length(header), nrow = 0))
 colnames(data_frame_for_region) <- header
 vector_of_numbers_of_rows <- rep(0, number_of_paths_to_data_files_for_region)
 for (i in 1:number_of_paths_to_data_files_for_region) {
  data_frame <- read.table(
   file = vector_of_paths_to_data_files_for_region[i],
   comment.char = ";"
  )
  colnames(data_frame) <- header
  data_frame_for_region <- rbind(data_frame_for_region, data_frame)
  vector_of_numbers_of_rows[i] = nrow(data_frame)
 }
 data_set_for_region <- list(
  data_frame = data_frame_for_region,
  vector_of_numbers_of_rows = vector_of_numbers_of_rows
 )
 return(data_set_for_region)
}
data_set_for_region <- generate_data_set_for_region(vector_of_paths_to_data_files_for_region)
print(head(data_set_for_region$data_frame))
print(nrow(data_set_for_region$data_frame))
print(data_set_for_region$vector_of_numbers_of_rows)

data_frame_to_plot <- data_set_for_region$data_frame[, c("X", "Y", "B1", "B2", "B3")]
colnames(data_frame_to_plot) <- c("x", "y", "r", "g", "b")

library(TomLeversRPackage)
the_ggplot <- plot_pixels(data_frame_to_plot)
print(the_ggplot)