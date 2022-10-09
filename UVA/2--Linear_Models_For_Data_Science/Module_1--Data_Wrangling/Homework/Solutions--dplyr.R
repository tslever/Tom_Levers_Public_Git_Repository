##################
##dplyr approach##
##################

library(tidyverse)

Covid<-read.csv("UScovid.csv", header=TRUE)

#######
##Q1a##
#######

##create data frame by criteria
latest<-Covid %>% 
  filter(date =="2021-06-03" & county != "Unknown") %>% 
  select(-c(1,4)) %>% 
  arrange(county,state)

head(latest)

#######
##Q1b##
#######

##calculate death rate for each county and add to data frame
latest<-latest %>% 
  mutate(death.rate=round(deaths/cases * 100,2))

head(latest)

#######
##Q1c##
#######

##find 10 counties with highest number of cases
latest<-latest %>% 
  arrange(desc(cases))

latest[1:10,]

#######
##Q1d##
#######

##find 10 counties with highest number of deaths
latest<-latest %>% 
  arrange(desc(deaths))

latest[1:10,]

#######
##Q1e##
#######

##find 10 counties with highest death rates
latest<-latest %>% 
  arrange(desc(death.rate))

latest[1:10,]

#######
##Q1f##
#######

##consider counties with at least 100,000 cases
most.cases<- latest %>% 
  filter(cases >= 100000)

##find counties with 10 highest death rates, with at least 100,000 cases
most.cases<-most.cases %>% 
  arrange(desc(death.rate))

most.cases[1:10,]

#######
##Q1g##
#######

##find numbers for Albemarle county, VA
latest %>% 
  filter(county == "Albemarle", state == "Virginia")

##find numbers for Charlottesville, VA
latest %>% 
  filter(county == "Charlottesville city", state == "Virginia")

#######
##Q2a##
#######

##since we are interested in state level data, we can 
##include counties which are unknown since the states
##are known
counties.latest<-Covid %>% 
  filter(date=="2021-06-03") %>% 
  select(-c(1,4))

##total cases by state
state1<- counties.latest%>%
  group_by(state)%>%
  summarize(Cases=sum(cases,na.rm=T))

##total deaths by state
state2<- counties.latest%>%
  group_by(state)%>%
  summarize(Deaths=sum(deaths,na.rm=T))

##merge cases and deaths into a data frame 
state.level<-state1 %>% 
  inner_join(state2, by="state") %>% 
  arrange(state)

head(state.level)

#######
##Q2b##
#######

##calculate death rate and add to data frame
state.level<- state.level %>% 
  mutate(state.rate=round(Deaths/Cases * 100, 2))

head(state.level)

#######
##Q2c##
#######

##VA's death rate
state.level %>% 
  filter(state=="Virginia")

#######
##Q2d##
#######

##PR's death rate
state.level %>% 
  filter(state=="Puerto Rico")

#######
##Q2e##
#######

##10 highest death rates
state.level<-state.level %>% 
  arrange(desc(state.rate))

state.level[1:10,]

#######
##Q2f##
#######

##10 lowest death rates
state.level<-state.level %>% 
  arrange(state.rate)

state.level[1:10,]

#######
##Q2g##
#######

write.csv(state.level, file="stateCovid.csv", row.names = TRUE)