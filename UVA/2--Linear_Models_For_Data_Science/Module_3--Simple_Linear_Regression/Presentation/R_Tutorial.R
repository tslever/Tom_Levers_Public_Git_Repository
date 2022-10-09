library(tidyverse)

## store data file with the variable name Data
Data<-read.csv("rocket.csv", header=TRUE)
head(Data)

##remove first column
Data<-Data[,-1]
##rename the remaining 2 columns
names(Data)<-c("Strength", "Age")
head(Data)

###############
##scatterplot##
###############

ggplot(Data, aes(x=Age,y=Strength))+
  geom_point()+
  labs(x="Age (Weeks)", y="Strength (PSI)", title="Scatterplot of Strength against Age")

plot(Data$Age, Data$Strength, xlab="Age (Weeks)", ylab="Strength (PSI)", main="Scatterplot of Strength against Age")

#############
##Save plot##
#############

jpeg("plot.jpg")
ggplot(Data, aes(x=Age,y=Strength))+
  geom_point()+
  labs(x="Age (Weeks)", y="Strength (PSI)", title="Scatterplot of Strength against Age")
dev.off()

#######
##SLR##
#######

##Fit a regression model
result<-lm(Strength~Age, data=Data)
summary(result)

##see what can be extracted from result
names(result)

##extract residuals
result$residuals

###############
##ANOVA table##
###############

anova.tab<-anova(result)
anova.tab

names(anova.tab)

SST<-sum(anova.tab$"Sum Sq")
SST

anova.tab$"Sum Sq"[1]/SST