################################
###Traditional Data Wrangling###
################################

Data<-read.csv("ClassDataPrevious.csv", header=TRUE)

#########################################
##view specific row(s) and/or column(s)##
#########################################

Data[1,2] ##row index first, then column index
Data[c(1,3,4),c(1,5,8)]

##view specific column(s)
Data$Year ##or
Data[,1] ##or
Data[,-c(2:8)]

##view specific row(s)
Data[c(1,3),]
Data[10:20,]

#######################################
##select observations by condition(s)##
#######################################

which(Data$Sport=="Soccer")
Data[which(Data$Sport=="Soccer"),]
SoccerPeeps<-Data[which(Data$Sport=="Soccer"),]
dim(SoccerPeeps)
SoccerPeeps_2nd<-Data[which(Data$Sport=="Soccer" & Data$Year=="Second"),]
dim(SoccerPeeps_2nd)
Sleepy<-Data[which(Data$Sleep>8),]
Sleepy_or_Soccer<-Data[which(Data$Sport=="Soccer" | Data$Sleep>8),]

#############################
##change names of column(s)##
#############################
names(Data)[7]<-"Comp"
names(Data)[c(1,7)]<-c("Yr","Computer")

################
##missing data##
################

##find which rows have missing data
is.na(Data) 
Data[!complete.cases(Data),]

##remove observations with missing values. CAUTION!
Data_nomiss<-na.omit(Data) ##or
Data_nomiss2<-Data[complete.cases(Data),]

########################
##Summarize a variable##
########################

##find means of numeric columns
apply(Data[,c(2,4,6,8)],2,mean)
apply(Data[,c(2,4,6,8)],2,mean,na.rm=T)
##means are suspiciously high, perhaps due to data entry errors
##find medians instead
apply(Data[,c(2,4,6,8)],2,median,na.rm=T)

###################################
##Summarize a variable, by groups##
###################################

##find median sleep by Yr
tapply(Data$Sleep,Data$Yr,median,na.rm=T)
##change the ordering of the factor Yr that is more pleasing
Data$Yr<-factor(Data$Yr, levels=c("First","Second","Third","Fourth"))
levels(Data$Yr)
tapply(Data$Sleep,Data$Yr,median,na.rm=T) ##much nicer

##find median sleep by Yr and Computer 
tapply(Data$Sleep,list(Data$Yr,Data$Computer),median,na.rm=T)

#######################################################
##Create a new variable based on existing variable(s)##
#######################################################

##convert Sleep to minutes
Sleep_mins<-Data$Sleep * 60

##create sleep deprived variable: yes if sleep less than 7 hours, 
##no otherwise
deprived<-ifelse(Data$Sleep<7, "yes", "no")

##create courseload category: light if 3 courses or less, 
##regular if 4 or 5 courses, heavy if more than 5 courses 
CourseLoad<-cut(Data$Courses, breaks = c(-Inf, 3, 5, Inf), labels = c("light", "regular", "heavy"))

##collapse classes of Yr: 1st and 2nd years to under,
##3rd and 4th years to upper
levels(Data$Yr)
new.levels<-c("und", "und", "up","up")
Year2<-factor(new.levels[Data$Yr])

###################################
##Combine data frames and vectors##
###################################

##add newly created variables as new columns to Data
Data<-data.frame(Data,Sleep_mins,deprived,CourseLoad,Year2)
Data2<-cbind(Data,Sleep_mins,deprived,CourseLoad,Year2)

##combine data frames with different rows, same columns
dat1<-Data[1:3,1:3]
dat3<-Data[6:8,1:3]
res.dat2<-rbind(dat1,dat3)

#################################
##export data frame to .csv file##
#################################

write.csv(Data, file="newdata.csv", row.names = TRUE)

#######################
##sort data by column##
#######################

##sort in ascending order by Age
Data_by_age<-Data[order(Data$Age),]

##sort in descending order by Age
Data_by_age_des<-Data[order(-Data$Age),]

##sort in ascending order by Age and then Sleep
Data_by_age_sleep<-Data[order(Data$Age, Data$Sleep),]


