vector_of_paths_to_data_files_for_region <- c(
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir078_ROI_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir078_ROI_NON_Blue_Tarps.txt"
)
source("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/generate_data_set.R")
data_set_for_region <- generate_data_set(vector_of_paths_to_data_files_for_region)
print(head(data_set_for_region$data_frame))
print(nrow(data_set_for_region$data_frame))
print(data_set_for_region$vector_of_numbers_of_rows)
data_frame_to_plot <- data_set_for_region$data_frame[, c("X", "Y", "B1", "B2", "B3")]
colnames(data_frame_to_plot) <- c("x", "y", "r", "g", "b")
the_ggplot <- TomLeversRPackage::plot_pixels(data_frame_to_plot)
print(the_ggplot)