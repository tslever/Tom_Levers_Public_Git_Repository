#############################
##Data Wrangling with dplyr##
#############################

##load tidyverse or dplyr package

##library(dplyr) or
library(tidyverse) 

##loading tidyverse automatically loads dplyr

Data<-read.csv("ClassDataPrevious.csv", header=TRUE)

#########################################
##view specific row(s) and/or column(s)##
#########################################

##view specific column(s)
select(Data,Year)

Data%>%
  select(Year)

##select multiple columns
Data%>%
  select(Year,Sleep)

#######################################
##select observations by condition(s)##
#######################################

filter(Data, Sport=="Soccer")

SoccerPeeps<-Data%>%
  filter(Sport=="Soccer")

SoccerPeeps_2nd<-Data%>%
  filter(Sport=="Soccer" & Year=="Second")

Sleepy<-Data%>%
  filter(Sleep>8)

Sleepy_or_Soccer<-Data%>%
  filter(Sport=="Soccer" | Sleep>8)

#############################
##change names of column(s)##
#############################

Data<-Data%>%
  rename(Yr=Year, Comp=Computer)

################
##missing data##
################

##find which rows have missing data
is.na(Data) 
Data[!complete.cases(Data),]

##remove observations with missing values. CAUTION!
Data_nomiss<-na.omit(Data) ##or
Data_nomiss2<-Data[complete.cases(Data),]

##Obs 103 has a missing value for Sleep,
##Obs 206 has a missing value for Lunch

########################
##Summarize a variable##
########################

##find means of numeric columns
Data%>%
  summarize(mean(Sleep,na.rm = T),mean(Courses),mean(Age),mean(Lunch,na.rm = T))

Data%>%
  summarize(avgSleep=mean(Sleep,na.rm = T),avgCourse=mean(Courses),avgAge=mean(Age),avgLun=mean(Lunch,na.rm = T))
##means are suspiciously high, perhaps due to data entry errors
##find medians instead
Data%>%
  summarize(medSleep=median(Sleep,na.rm = T),medCourse=median(Courses),medAge=median(Age),medLun=median(Lunch,na.rm = T))

###################################
##Summarize a variable, by groups##
###################################

Data%>%
  group_by(Yr)%>%
  summarize(medSleep=median(Sleep,na.rm=T))

##change the ordering of the factor Yr that is more pleasing
Data<- Data%>%
  mutate(Yr = Yr%>%
           fct_relevel(c("First","Second","Third","Fourth"))
         )

Data%>%
  group_by(Yr)%>%
  summarize(medSleep=median(Sleep,na.rm=T))

##find median sleep by Yr and Computer 
Data%>%
  group_by(Yr,Comp)%>%
  summarize(medSleep=median(Sleep,na.rm=T))

#######################################################
##Create a new variable based on existing variable(s)##
#######################################################

##convert Sleep to minutes and add new variable
Data<-Data%>%
  mutate(Sleep_mins = Sleep*60)

##create sleep deprived variable: yes if sleep less than 7 hours, 
##no otherwise
Data<-Data%>%
  mutate(deprived=ifelse(Sleep<7, "yes", "no"))


##create courseload category: light if 3 courses or less, 
##regular if 4 or 5 courses, heavy if more than 5 courses 
Data<-Data%>%
  mutate(CourseLoad=cut(Courses, breaks = c(-Inf, 3, 5, Inf), labels = c("light", "regular", "heavy")))

##collapse classes of Yr: 1st and 2nd years to under,
##3rd and 4th years to upper
Data<-Data%>%
  mutate(UpUnder=fct_collapse(Yr,under=c("First","Second"),up=c("Third","Fourth")))
################################
##Merge data frames and vectors##
################################


##merge data frames with different rows, same columns
dat1<-Data[1:3,1:3]
dat3<-Data[6:8,1:3]
res.dat2<-bind_rows(dat1,dat3)

##use bind_cols for cbind()

#################################
##export data frame to .csv file##
#################################

write.csv(Data, file="newdata.csv", row.names = TRUE)

#######################
##sort data by column##
#######################

##sort in ascending order by Age
Data_by_age<-Data%>%
  arrange(Age)

##sort in descending order by Age
Data_by_age_des<-Data%>%
  arrange(desc(Age))

##sort in ascending order by Age and then Sleep
Data_by_age_sleep<-Data%>%
  arrange(Age,Sleep)


