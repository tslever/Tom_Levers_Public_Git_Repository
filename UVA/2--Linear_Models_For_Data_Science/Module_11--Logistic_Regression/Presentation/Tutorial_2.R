################
##Example 2#####
##grouped data##
##dose##########
################

library(tidyverse)
Data2<-read.table("dose.txt", header=T)

#######
##EDA##
#######

##calculate proportion that died
Data2$prop<-Data2$died/Data2$size
##calculate log odds that died
Data2$log.odds<-log(Data2$prop/(1-Data2$prop))

##plot log odds against dose
ggplot(Data2, aes(x=logdose,y=log.odds))+
  geom_point()+
  labs(x="logdose", y="log odds", 
       title="Scatterplot of Log Odds of Death against Dose (log)")

#########
##model##
#########

##fit logistic regression with grouped data
result2<-glm(prop~logdose, family="binomial", weights=size, data=Data2)

#############
##GOF tests##
#############

N<-dim(Data2)[1]
p<-2
##pearson chi square goodness of fit
pearson<-residuals(result2,type="pearson")
##calculate the test stat
X2<-sum(pearson^2)
X2
##p-value
1-pchisq(X2,N-p)

##compare with deviance goodness of fit
##calculate the test stat
result2$deviance
##p-value
1-pchisq(result2$deviance,N-p)
