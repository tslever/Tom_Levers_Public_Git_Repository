vector_of_paths_to_data_files <- c(
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir057_ROI_NON_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir067_ROI_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir067_ROI_NOT_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir069_ROI_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir069_ROI_NOT_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir078_ROI_Blue_Tarps.txt",
 "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/orthovnir078_ROI_NON_Blue_Tarps.txt"
)
source("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/generate_data_set.R")
holdout_data_set <- generate_data_set(vector_of_paths_to_data_files)
print(head(holdout_data_set$data_frame))
print(nrow(holdout_data_set$data_frame))
print(holdout_data_set$vector_of_numbers_of_rows)
holdout_data_frame <- holdout_data_set$data_frame[, c("Class", "B1", "B2", "B3")]
colnames(holdout_data_frame) <- c("Class", "Red", "Green", "Blue")
print(head(holdout_data_frame))
print(nrow(holdout_data_frame))
write.csv(holdout_data_frame, "C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project/Holdout_Data/Holdout_Data_Frame.csv", row.names = FALSE)
print(nrow(holdout_data_frame[holdout_data_frame$Class == "Not Blue Tarp",]))
print(nrow(holdout_data_frame[holdout_data_frame$Class == "Blue Tarp",]))