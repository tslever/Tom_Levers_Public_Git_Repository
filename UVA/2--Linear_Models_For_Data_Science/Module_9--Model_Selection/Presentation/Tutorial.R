library(leaps)

Data<-mtcars

############################
##all possible regressions##
############################

##perform all possible regressions (nbest=1)
allreg <- regsubsets(mpg ~., data=Data, nbest=1)
summary(allreg)

##perform all possible regressions (nbest=2)
allreg2 <- regsubsets(mpg ~., data=Data, nbest=2)
summary(allreg2)

##see what can be extracted
names(summary(allreg2))

##find model with best according to different criteria
which.max(summary(allreg2)$adjr2)
which.min(summary(allreg2)$cp)
which.min(summary(allreg2)$bic)

##find coefficients and predictors of model with best
##adj r2, cp, bic
coef(allreg2, which.max(summary(allreg2)$adjr2))
coef(allreg2, which.min(summary(allreg2)$cp))
coef(allreg2, which.min(summary(allreg2)$bic))

#########################################################
##Forward selection, backward elimination, stepwise reg##
#########################################################

##intercept only model
regnull <- lm(mpg~1, data=Data)
##model with all predictors
regfull <- lm(mpg~., data=Data)

##forward selection, backward elimination, and stepwise regression
step(regnull, scope=list(lower=regnull, upper=regfull), direction="forward")
step(regfull, scope=list(lower=regnull, upper=regfull), direction="backward")
step(regnull, scope=list(lower=regnull, upper=regfull), direction="both")
