#############
##Example 1##
#############

library(tidyverse)

## store data file with the variable name Data
Data<-read.csv("electricity.csv", header=TRUE)
head(Data)

##remove first column
Data<-Data[,-1]
##rename the remaining 2 columns
names(Data)<-c("Usage","Demand")
head(Data)

#####################################
##Scatterplot for model diagnostics##
#####################################

##scatterplot, and overlay regression line
ggplot(Data, aes(x=Usage,y=Demand))+
  geom_point()+
  geom_smooth(method = "lm", se=FALSE)+
  labs(x="Energy Usage (kWh)", y="Energy Demand (kW)", title="Scatterplot of Energy Demand vs Energy Usage")

##base R
plot(Data$Usage, Data$Demand, xlab="Energy Usage (kWh)", ylab="Energy Demand (kW)", main="Scatterplot of Energy Demand vs Energy Usage")
abline(lm(Demand~Usage, data=Data))

#######################################
##Residual plot for model diagnostics##
#######################################

##residual plot may be easier to visualize

##Fit a regression model
result<-lm(Demand~Usage, data=Data)

##store fitted y & residuals
yhat<-result$fitted.values
res<-result$residuals

##add to data frame
Data<-data.frame(Data,yhat,res)

##residual plot
ggplot(Data, aes(x=yhat,y=res))+
  geom_point()+
  geom_hline(yintercept=0, color="red")+
  labs(x="Fitted y", y="Residuals", title="Residual Plot")

plot(yhat,res, xlab="Fitted y", ylab="Residuals", main="Residual Plot")
abline(h=0, col="red")

##notice fanning out pattern. Need to transform y.

##########################
##Box Cox to transform y##
##########################

library(MASS) ##to use boxcox function

boxcox(result)

##adjust lambda for better visualization

boxcox(result, lambda = seq(0, 1, 1/10)) 
##use lambda=0.5 to squareroot response

##transform y and then regress ystar on x
ystar<-(Data$Demand)^0.5
Data<-data.frame(Data,ystar)
result.ystar<-lm(ystar~Usage, data=Data)

##store fitted y & residuals
yhat2<-result.ystar$fitted.values
res2<-result.ystar$residuals

##add to data frame
Data<-data.frame(Data,yhat2,res2)

##residual plot with ystar
ggplot(Data, aes(x=yhat2,y=res2))+
  geom_point()+
  geom_hline(yintercept=0, color="red")+
  labs(x="Fitted y", y="Residuals", title="Residual Plot with ystar")

boxcox(result.ystar)

##Looks good. No need for any more transformations on x or y

#########################
##ACF Plot of residuals##
#########################

acf(res2, main="ACF Plot of Residuals with ystar")

########################
##QQ plot of residuals##
########################

qqnorm(res2)
qqline(res2, col="red")

result.ystar
