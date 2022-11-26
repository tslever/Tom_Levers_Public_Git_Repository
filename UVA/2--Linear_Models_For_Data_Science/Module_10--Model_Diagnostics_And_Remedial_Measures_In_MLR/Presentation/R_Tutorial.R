Data<-read.table("delivery.txt", header=TRUE)
result<-lm(Delivery~Number+Distance, data=Data)

################################
##different types of residuals##
################################

##residuals, e_i
res<-result$residuals 

##standardized residuals, d_i
standard.res<-res/summary(result)$sigma

##studentized residuals, r_i
student.res<-rstandard(result) 

##externally studentized residuals, t_i
ext.student.res<-rstudent(result) 

res.frame<-data.frame(res,standard.res,student.res,ext.student.res)

res.frame[9,]

par(mfrow=c(1,3))
plot(result$fitted.values,standard.res,main="Standardized Residuals", ylim=c(-4.5,4.5))
plot(result$fitted.values,student.res,main="Studentized Residuals", ylim=c(-4.5,4.5))
plot(result$fitted.values,ext.student.res,main="Externally  Studentized Residuals", ylim=c(-4.5,4.5))

#############################
##identify outlier with t_i##
#############################

##critical value using Bonferroni procedure
n<-dim(Data)[1]
p<-3
crit<-qt(1-0.05/(2*n), n-1-p)

##identify
ext.student.res[abs(ext.student.res)>crit]

#########################
##outlier in predictors##
#########################

##leverages
lev<-lm.influence(result)$hat 

##identify high leverage points
lev[lev>2*p/n]

############################
##influential observations##
############################

##cooks distance
COOKS<-cooks.distance(result)
COOKS[COOKS>qf(0.5,p,n-p)]

##dffits
DFFITS<-dffits(result)
DFFITS[abs(DFFITS)>2*sqrt(p/n)]

##dfbetas
DFBETAS<-dfbetas(result)
abs(DFBETAS)>2/sqrt(n)

