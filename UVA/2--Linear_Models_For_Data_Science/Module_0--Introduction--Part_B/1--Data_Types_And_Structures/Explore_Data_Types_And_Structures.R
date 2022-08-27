##############
##Data Types##
##############

##numeric

##2 ways to assign 30 to the object called age
age<-30 ##recommended way
age=30

class(age) ##or
is.numeric(age) 

##character

today<-"Monday"

today<-Monday

class(today) ##or
is.character(today) 

##logical 

yes_or_no<-TRUE ##or
yes_or_no2<-T

is.logical(yes_or_no)
class(yes_or_no2)

##examples with conditions
(2+1)==3
(2+1)!=3
(5+7)>12
(5+7)>=12
(2+1==3)&(2+1==4)
(2+1==3)|(2+1==4)

##create numeric vector with c()
Age<-c(30,18,19,20,21) 
Age

##create character vector
Day_of_week<-c("Monday","Wednesday","Saturday","Sunday","Monday") 
Day_of_week

##create logical vector
logic<-c(TRUE,T,FALSE,F,T) 
logic

##check type
is.vector(Age)
is.numeric(Age)
is.character(Age)
is.logical(Age)

##coercion
try<-c(3.1, "Tuesday")
class(try)
try2<-c(T, "Friday")
class(try2)
try3<-c(3.1, F)
class(try3)

##matrices

mat.A<-matrix(c(4,3,6,1,2,1), nrow=3, ncol=2)
mat.A

##array

arr.A<-array(0, dim=c(2,3,2))
arr.A

##data frame
###combining vectors (of the same length) to a data frame.
Data<-data.frame(Age,Day_of_week,logic)
Data

##lists

big<-list(mat.A, Day_of_week, Age, age, logic)
big

##give names to object
names(Data)<-c("Friends", "Pets", "Logic?")
Data

##extract a specific element in vector
Age[3] ##third element

##extract a specific element in matrix
mat.A[2,1] ##row 2, column 1

##extract object 1 and 2 from list
big[[1]]
big[[2]]

##number of elements in vector
length(Day_of_week)

##number of rows and cols in matrix, array, data frame
dim(mat.A)
nrow(mat.A)
ncol(mat.A)
dim(arr.A)

##convert vector type
Age2<-as.character(Age)
Age2
mean(Age)
mean(Age2) ##error

##factors
DOW<-factor(Day_of_week)
DOW
DOW<-factor(Day_of_week,levels = c("Sunday","Monday","Wednesday","Saturday"))
DOW

##convert numeric to factor
Age_ord<-factor(Age) 
Age_ord
##R orders numeric from smallest to largest by default

##convert factor to numeric.
Age3<-as.numeric(as.character(Age_ord))
Age3