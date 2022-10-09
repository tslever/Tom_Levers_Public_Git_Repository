##################
##dplyr approach##
##################

library(tidyverse)

students<-read.table("students.txt", header=TRUE) 

##############
##Question 1##
##############

##remove first column
students.tidy<-students %>% 
  select(-1)

##############
##Question 2##
##############

##number of students
nrow(students.tidy)

##############
##Question 3##
##############

##number of students with missing data
missing<-students.tidy[!complete.cases(students.tidy),]
nrow(missing)

##############
##Question 4##
##############

##find medians of numeric variables
students.tidy%>%
  summarize(medGPA=median(GPA,na.rm = T),medParty=median(PartyNum,na.rm = T),medBeer=median(DaysBeer,na.rm = T),medStudy=median(StudyHrs,na.rm = T))

##############
##Question 5##
##############

##mean of studyhrs by gender
students.tidy%>%
  group_by(Gender)%>%
  summarize(meanStudy=mean(StudyHrs,na.rm=T))

##SD of studyhrs by gender
students.tidy%>%
  group_by(Gender)%>%
  summarize(sdStudy=sd(StudyHrs,na.rm=T))

##############
##Question 6##
##############

##mean of studyhrs by gender
study.mean<-students.tidy%>%
  group_by(Gender)%>%
  summarize(meanStudy=mean(StudyHrs,na.rm=T))

##SD of studyhrs by gender
study.sd<-students.tidy%>%
  group_by(Gender)%>%
  summarize(sdStudy=sd(StudyHrs,na.rm=T))

##sample sizes of genders
sample.size<-students.tidy%>%
  count(Gender)

##t multipliers 
t.female<-qt(0.975,sample.size[1,2]-1)
t.male<-qt(0.975,sample.size[2,2]-1)

##lower and upper bound for CI, female
study.mean[1,2]-t.female*study.sd[1,2]/sqrt(sample.size[1,2])
study.mean[1,2]+t.female*study.sd[1,2]/sqrt(sample.size[1,2])

##lower and upper bound for CI, male
study.mean[2,2]-t.male*study.sd[2,2]/sqrt(sample.size[2,2])
study.mean[2,2]+t.male*study.sd[2,2]/sqrt(sample.size[2,2])

##############
##Question 7##
##############

##median studyhrs by gender and smoke
students.tidy%>%
  group_by(Gender,Smoke)%>%
  summarize(medStudy=median(StudyHrs,na.rm=T))

##############
##Question 8##
##############

##new variable PartyAnimal
students.tidy<-students.tidy%>%
  mutate(PartyAnimal= ifelse(PartyNum>8, "yes", "no"))

##############
##Question 9##
##############

##new variable GPA.cat
students.tidy<-students.tidy%>%
  mutate(GPA.cat= cut(GPA, breaks = c(-Inf, 3.0, 3.5, Inf), right=FALSE, labels = c("low", "moderate", "high")))

###############
##Question 10##
###############

##export data as .csv
write.csv(students.tidy, file="new_students.csv", row.names = TRUE)

###############
##Question 11##
###############

##number of students with the 3 criteria
fun.times<-students.tidy%>%
  filter(GPA<3.0 & PartyNum>8 & StudyHrs<15)
        
nrow(fun.times)
