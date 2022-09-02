######################################################
##Data Visualization 2: Single Quantitative Variable##
######################################################

library(tidyverse)

Data<-read.csv("ClassDataPrevious.csv", header=TRUE)

####################
##5 number summary##
####################

##5 number summary and mean
summary(Data$Age)

###########
##Boxplot##
###########

##Boxplot is a graphical representation of the 5 number summary

##generic boxplot
ggplot(Data, aes(y=Age))+
  geom_boxplot()

##change orientation of boxplot
ggplot(Data, aes(y=Age))+
  geom_boxplot()+
  coord_flip()

##change colors
ggplot(Data, aes(y=Age))+
  geom_boxplot(color="blue", outlier.color = "orange" )

##boxplots of Age across years
ggplot(Data, aes(x=Year, y=Age))+
  geom_boxplot(color="blue", outlier.color = "orange" )

##want to change order of Year to make more sense
Data$Year<-factor(Data$Year, levels=c("First","Second","Third","Fourth"))
levels(Data$Year)

ggplot(Data, aes(x=Year, y=Age))+
  geom_boxplot(color="blue", outlier.color = "orange" )

##Boxplot of age across year and computer
ggplot(Data, aes(x=Year, y=Age, fill=Computer))+
  geom_boxplot(color="blue", outlier.color = "orange" )

##violin plot instead of side by side box plots. Requires an x variable
ggplot(Data, aes(x=Year, y=Age))+
  geom_violin()

ggplot(Data, aes(x=Year, y=Age, fill=Computer))+
  geom_violin(color="blue")

##histogram
ggplot(Data,aes(x=Age))+
  geom_histogram()
##note warning message

##how to pick value for bins
sqrt(nrow(Data))

ggplot(Data,aes(x=Age))+
  geom_histogram(bins = 17,fill="blue",color="orange")

##density plot
ggplot(Data,aes(x=Age))+
  geom_density()

ggplot(Data,aes(x=Age))+
  geom_density(fill="blue",color="orange")
