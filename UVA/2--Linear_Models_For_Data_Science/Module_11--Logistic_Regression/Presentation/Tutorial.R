##################
##Example 1#######
##Ungrouped data##
##titanic#########
##################

library(tidyverse)
Data<-read.table("titanic.txt", header=TRUE)

#######
##EDA##
#######

##frequency table
table(Data$Survived)
##table with proportions
prop.table(table(Data$Survived))

##2 way table
mytab<-table(Data$Sex, Data$Survived)
mytab
##survival rate for each gender
prop.table(mytab, 1) 

##bar chart

ggplot(Data, aes(x=Sex, fill=Survived))+
  geom_bar(position = "fill")+
  labs(x="Gender", y="Proportion", title="Survival Rate by Gender")

is.numeric(Data$Survived)
##change Survived to factor for visuals
Data$Survived<-factor(Data$Survived)
levels(Data$Survived)
##notice 0 first, then 1
##give descriptive labels
levels(Data$Survived) <- c("No","Yes") 
levels(Data$Survived)

##Recreate bar chart 
ggplot(Data, aes(x=Sex, fill=Survived))+
  geom_bar(position = "fill")+
  labs(x="Gender", y="Proportion", title="Survival Rate by Gender")

##side by side boxplots
ggplot(Data, aes(x=Survived, y=Fare))+
  geom_boxplot()+
  labs(title="Dist of Fare Paid by Survival Status")

##density plots
ggplot(Data,aes(x=Fare, color=Survived))+
  geom_density()+
  labs(title="Density Plot of Fare Paid by Survival Status")

#############
##Fit model##
#############

##fit logistic regression using glm
result<-glm(Survived ~ Sex, family = "binomial", data=Data)
summary(result)

##full model
full<-glm(Survived ~ ., family = "binomial", data=Data)
summary(full)

##test if coefficients for all 3 predictors are 0
##test stat
TS<-full$null.deviance-full$deviance
##pvalue
1-pchisq(TS,3)

##test if additional predictors have coefficients equal to 0
##test stat
TS2<-result$deviance-full$deviance
##pvalue
1-pchisq(TS2,2)

