library(tidyverse)

Data<-read.table("wine.txt", header=TRUE, sep="")

##make sure Region is correctly recognized as a factor

class(Data$Region)

##convert Region to categorical
Data$Region<-factor(Data$Region) 
class(Data$Region) 
levels(Data$Region)

##Give names to the levels
levels(Data$Region) <- c("North", "Central", "Napa") 
levels(Data$Region)

##scatterplot of Quality against Flavor, 
##separated by Region
ggplot(Data, aes(x=Flavor, y=Quality, color=Region))+
  geom_point()+
  geom_smooth(method=lm, se=FALSE)+
  labs(title="Scatterplot of Wine Quality against Flavor, by Region")

##slopes almost parallel

##Set a different reference class
contrasts(Data$Region)
Data$Region<-relevel(Data$Region, ref = "Napa") 
contrasts(Data$Region)

##consider model with interactions 
##(when slopes are not parallel)
result<-lm(Quality~Flavor*Region, data=Data)
summary(result)

##fit regression with no interaction
reduced<-lm(Quality~Flavor+Region, data=Data)

##Partial F test for interaction terms
anova(reduced,result)
##go with reduced model with no interactions. 
##Not surprising given scatterplot

##additional assumption to check with categorical predictor. 
##Is the variance of the response variable constant 
##between all classes of the categorical predictor?
ggplot(Data, aes(x=Region, y=Quality))+
  geom_boxplot()+
  labs(title="Dist of Wine quality by Region")

##perform levene's test. 
##Null states the variances are equal for all classes. 
library(lawstat)
levene.test(Data$Quality,Data$Region)

##multiple comparisons. 
##Can be used when there is no interactions
library(multcomp)
pairwise<-glht(reduced, linfct = mcp(Region= "Tukey"))
summary(pairwise)

######################################################
##used in PDF for Bonferroni method, not in tutorial##
######################################################

reduced$coef

##obtain the variance-covariance matrix of the coefficients
vcov(reduced)

