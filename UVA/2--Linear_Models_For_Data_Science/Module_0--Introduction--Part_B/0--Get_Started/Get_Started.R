#####################
##Working Directory##
#####################

##get working directory
getwd()
###################################
##3 ways to set working directory##
###################################

##1: use setwd() function, I don't this at all
?setwd 

##2: click on Session, then Set Working Directory and then 
##Choose Directory

##3: create a shortcut for RStudio in the folder that you 
##want to use a working directory, then right click,
##choose properties, select the tab Shortcut, then delete
##everything in the box next to Start In. Now when you click
##on the shortcut, the folder the shortcut is in becomes the
##working directory. 

##read in data from .txt file file
titanic<-read.table("titanic.txt", header=TRUE) 

##read in data from .csv file
wcgs<-read.csv("wcgs.csv", header=TRUE) 

##R has some datasets built in
data()
##read in a dataframe that comes with R
women.df<-women 

##install tidyverse package

install.packages("tidyverse")

##after installing, need to load packages, 
##then can use functions from these packages
library(tidyverse)