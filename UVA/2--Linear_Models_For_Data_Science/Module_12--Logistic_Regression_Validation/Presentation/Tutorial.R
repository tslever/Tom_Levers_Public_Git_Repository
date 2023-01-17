library(ROCR)
Data<-read.table("titanic.txt", header=TRUE, sep="")

##set the random number generator so same results can be reproduced
set.seed(111)

##choose the observations to be in the training set. I am splitting the dataset into halves
sample<-sample.int(nrow(Data), floor(.50*nrow(Data)), replace = F)
train<-Data[sample, ] ##training data
test<-Data[-sample, ] ##test data

##use training data to fit logistic regression model with fare and gender as predictors
full<-glm(Survived~., family=binomial, data=train)
summary(full)

result<-glm(Survived~Sex+Fare, family="binomial", data=train)
summary(result)

##predicted survival rate for test data based on training data
preds<-predict(result,newdata=test, type="response")

##produce the numbers associated with classification table
rates<-prediction(preds, test$Survived)

##store the true positive and false postive rates
roc_result<-performance(rates,measure="tpr", x.measure="fpr")

##plot ROC curve and overlay the diagonal line for random guessing
plot(roc_result, main="ROC Curve for Titanic")
lines(x = c(0,1), y = c(0,1), col="red")

##compute the AUC
auc<-performance(rates, measure = "auc")
auc@y.values

##confusion matrix. Actual values in the rows, predicted classification in cols
table(test$Survived, preds>0.5)

table(test$Survived, preds>0.7)




