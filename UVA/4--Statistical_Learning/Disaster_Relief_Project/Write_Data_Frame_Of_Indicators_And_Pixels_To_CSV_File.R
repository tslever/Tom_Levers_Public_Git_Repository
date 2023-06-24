setwd("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/UVA/4--Statistical_Learning/Disaster_Relief_Project")
data_frame_of_classes_and_pixels <- read.csv(
 file = "Data_Frame_Of_Classes_And_Pixels.csv"
)

number_of_observations <- nrow(data_frame_of_classes_and_pixels)
column_of_indicators <- rep(0, number_of_observations)
condition <- data_frame_of_classes_and_pixels$Class == "Blue Tarp"
column_of_indicators[condition] <- 1
factor_of_indicators <- factor(column_of_indicators)
data_frame_of_indicators_and_pixels <- data.frame(
 Indicator = factor_of_indicators,
 Red = data_frame_of_classes_and_pixels$Red,
 Green = data_frame_of_classes_and_pixels$Green,
 Blue = data_frame_of_classes_and_pixels$Blue
)
data_frame_of_indicators_and_pixels <- data.frame(ID = seq(from = 1, to = number_of_observations, by = 1), data_frame_of_indicators_and_pixels)
#vector_of_random_indices <- sample(1:number_of_observations)
#data_frame_of_indicators_and_pixels <- data_frame_of_indicators_and_pixels[vector_of_random_indices, ]
#data_frame_of_indicators_and_pixels$Red <- data_frame_of_indicators_and_pixels$Red / 255
#data_frame_of_indicators_and_pixels$Green <- data_frame_of_indicators_and_pixels$Green / 255
#data_frame_of_indicators_and_pixels$Blue <- data_frame_of_indicators_and_pixels$Blue / 255
write.csv(x = data_frame_of_indicators_and_pixels, file = paste(getwd(), "/Data_Frame_Of_Indicators_And_Pixels.csv", sep = ""), row.names = FALSE)