Data<-read.table("delivery.txt", header=TRUE)

pairs(Data, lower.panel = NULL)

##Fit MLR model, using + in between predictors
result<-lm(Delivery~Number+Distance, data=Data)
##or
result<-lm(Delivery~., data=Data)

summary(result)

##CI for coefficients
confint(result,level = 0.95)

##Find CI for mean response and PI for a response for particular values of the predictors
newdata<-data.frame(Number=20, Distance=200)

predict(result, newdata, level=0.95, interval="confidence")
predict(result, newdata, level=0.95, interval="prediction")

############################
##End of tutorial document##
############################

##Practice model diagnostics and transformation
##on your own, then check below

##create residual plot
yhat<-result$fitted.values
res<-result$residuals
Data<-data.frame(Data,yhat,res)

ggplot(Data, aes(x=yhat,y=res))+
  geom_point()+
  geom_hline(yintercept=0, color="red")+
  labs(x="Fitted y", y="Residuals", title="Residual Plot")

##Looks curved. May need to transform at least one of
##the predictors. We will cover transforming predictors
##in MLR in Module 10
##Vertical spread looks to be increasing. Verify with
##Box Cox plot

library(MASS)

boxcox(result)
boxcox(result, lambda = seq(0,1,1/10))
##squareroot response

ystar<-(Data$Delivery)^0.5

Data<-data.frame(Data,ystar)

##refit regression with ystar
result.ystar<-lm(ystar~Number+Distance, data=Data)

##store fitted y & residuals
yhat2<-result.ystar$fitted.values
res2<-result.ystar$residuals

##add to data frame
Data<-data.frame(Data,yhat2,res2)

##residual plot
ggplot(Data, aes(x=yhat2,y=res2))+
  geom_point()+
  geom_hline(yintercept=0, color="red")+
  labs(x="Fitted y", y="Residuals", title="Residual Plot")
