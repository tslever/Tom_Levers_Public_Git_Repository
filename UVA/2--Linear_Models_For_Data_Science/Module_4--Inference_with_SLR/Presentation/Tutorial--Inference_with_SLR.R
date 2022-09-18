## store data file with the variable name Data
Data<-read.csv("rocket.csv", header=TRUE)
head(Data)

##remove first column
Data<-Data[,-1]
##rename the remaining 2 columns
names(Data)<-c("Strength", "Age")
head(Data)

##Fit a regression model
result<-lm(Strength~Age, data=Data)
summary(result)

##to produce 95% CIs for all regression coefficients
confint(result,level = 0.95)

##to produce 95% CI for the mean response when x=10, 
##and the 95% PI for the response of an observation when x=10
newdata<-data.frame(Age=10)
predict(result,newdata,level=0.95, interval="confidence")
predict(result,newdata,level=0.95, interval="prediction")

