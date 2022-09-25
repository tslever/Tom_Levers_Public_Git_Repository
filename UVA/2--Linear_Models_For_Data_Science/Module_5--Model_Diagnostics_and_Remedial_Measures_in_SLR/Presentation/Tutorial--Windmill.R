#############
##Example 2##
#############

library(tidyverse)

## store data file with the variable name Data
Data<-read.table("windmill.txt", header=TRUE)
head(Data)

#####################################
##Scatterplot for model diagnostics##
#####################################

##scatterplot, and overlay regression line
ggplot(Data, aes(x=wind,y=output))+
  geom_point()+
  geom_smooth(method = "lm", se=FALSE)+
  labs(x="Wind Velocity", y="DC Output", title="Scatterplot of DC Output Against Wind Velocity")

##Looks non linear. For wind velocity between 4 and 8, plots are above the regression line.
##Otherwise, the plots are below the regression line.
##Assumption 1 not met.
##Assumption 2 difficult to see in this picture

#######################################
##Residual plot for model diagnostics##
#######################################

##residual plot may be easier to visualize

##Fit a regression model
result<-lm(output~wind, data=Data)

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

##Curved pattern. So non linear relationship. 
##Assumption 1 not met.
##Vertical spread is constant though. 
##Assumption 2 met. 
##So no need to transform y, need to transform x. 

##Based on scatterplot, try inverse transform
##Log transform or squareroot transform could be tried as well
xstar<-1/(Data$wind)

Data<-data.frame(Data,xstar)

##regress y on xstar
result.xstar<-lm(output~xstar, data=Data)

yhat2<-result.xstar$fitted.values
res2<-result.xstar$residuals

##add to data frame
Data<-data.frame(Data,yhat2,res2)

##residual plot with xstar
ggplot(Data, aes(x=yhat2,y=res2))+
  geom_point()+
  geom_hline(yintercept=0, color="red")+
  labs(x="Fitted y", y="Residuals", title="Residual Plot with xstar")

##improvement. Residuals more evenly scattered across horizontal line
##Vertical spread appears constant.
##Assumptions 1 and 2 both met.

##confirm assumption 2 with Box Cox plot
library(MASS)

boxcox(result.xstar)
##1 inside interval, so no need to transform y. 

#########################
##ACF Plot of residuals##
#########################

acf(res2, main="ACF Plot of Residuals with xstar")
##uncorrelated residuals

########################
##QQ plot of residuals##
########################

qqnorm(res2)
qqline(res2, col="red")


