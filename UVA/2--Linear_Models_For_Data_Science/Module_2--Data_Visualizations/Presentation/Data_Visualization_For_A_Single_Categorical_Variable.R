#####################################################
##Data Visualization 1: Single Categorical Variable##
#####################################################

library(tidyverse)

Data<-read.csv("ClassDataPrevious.csv", header=TRUE)

##########
##Tables##
##########

##Table is a way to numerically summarize categorical variable

##Create a table to count the number of observations in each Year
table(Data$Year)

##want to change order of Year to make more sense
Data$Year<-factor(Data$Year, levels=c("First","Second","Third","Fourth"))
levels(Data$Year)
mytab<-table(Data$Year)
mytab

##convert table from count to proportion
prop.table(mytab)

##convert table from count to percent
prop.table(mytab) *100

##round to two decimal places
round(prop.table(mytab) * 100, 2)

##############
##Bar charts##
##############

##Bar chart is used to graphically summarize categorical variables

##generic bar chart
ggplot(Data, aes(x=Year))+
  geom_bar()

##change the orientation of the bar chart
ggplot(Data, aes(x=Year))+
  geom_bar()+
  coord_flip()

##change color of bars
ggplot(Data, aes(x=Year))+
  geom_bar(fill="blue")

##change color of outline of bars
ggplot(Data, aes(x=Year))+
  geom_bar(fill="blue",color="orange")

##add theme to change how the text of the x axis is re-oriented.
ggplot(Data, aes(x=Year))+
  geom_bar()+
  theme(axis.text.x = element_text(angle = 90))

##change the labels of the axes and title
ggplot(Data, aes(x=Year))+
  geom_bar()+
  theme(axis.text.x = element_text(angle = 90))+
  labs(x="Year", y="Number of Students", title="Dist of Years")

##change the position of the title in theme
ggplot(Data, aes(x=Year))+
  geom_bar()+
  theme(axis.text.x = element_text(angle = 90), plot.title = element_text(hjust = 0.5))+
  labs(x="Year", y="Number of Students", title="Dist of Years")

##create a bar chart where the vertical axis is
##proportion instead of count

##create a new data frame that contains the percentages for Year
newData<-Data%>%
  group_by(Year)%>%
  summarize(Counts=n())%>%
  mutate(Percent=Counts/nrow(Data))

newData

##then create a new bar chart, by adding some extra arguments in aes and geom_bar
ggplot(newData, aes(x=Year, y=Percent))+
  geom_bar(stat="identity")+
  theme(axis.text.x = element_text(angle = 90), plot.title = element_text(hjust = 0.5))+
  labs(x="Year", y="Percent of Students", title="Dist of Years")
