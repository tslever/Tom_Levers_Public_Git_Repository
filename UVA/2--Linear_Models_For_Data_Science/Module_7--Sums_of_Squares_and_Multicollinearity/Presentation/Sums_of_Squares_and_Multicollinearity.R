Data <- read.table("mileage.txt", header=TRUE)

##regress gas mileage against 4 predictors
result<-lm(y~x1+x2+x6+x10, data=Data)
summary(result)

##fit the reduced model with just displacement
reduced<-lm(y~x1, data=Data)

##perform the partial F test to see if we can drop the last 3 predictors
anova(reduced,result)

##another way to perform the same partial F test
anova(result)

##calculate the F statistic
F0<-((6.55+12.04+3.37)/3)/(259.86/27)
F0

##find the p-value for this F statistic
1-pf(F0,3,27)

##find the critical value for the F(3,27) distribution at 0.05 sig level
qf(0.95,3,27)

##pairwise correlation for the 4 predictors
cor(Data[,c(2,3,7,11)])

##vif function found in faraway package
library(faraway)

##VIFs for the regression model with 4 predictors
vif(result)

##another way to calculate the VIF
result2<-lm(x1~x2+x6+x10, data=Data)
summary(result2)

1/(1-0.9496)