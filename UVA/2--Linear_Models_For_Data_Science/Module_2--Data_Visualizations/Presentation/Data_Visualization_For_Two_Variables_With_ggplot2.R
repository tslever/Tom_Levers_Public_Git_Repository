###################################
##Data Visualization 3: Bivariate##
###################################

library(tidyverse)
library(gapminder)

View(gapminder)
##notice how data were collected over several years

##we want to focus on most recent data, from year 2007
Data<-gapminder%>%
  filter(year==2007)

#########################
##Side by side boxplots## 
########################

##useful to compare a quantitative variable across categorical 
##variable(s)

##Life expectancies across continents
ggplot(Data, aes(x=continent, y=lifeExp))+
  geom_boxplot(fill="Blue")+
  labs(x="Continent", y="Life Exp", title="Dist of Life Expectancies by Continent")

##Life expectancies across continents & years
ggplot(gapminder, aes(x=continent, y=lifeExp, fill=as.factor(year)))+
  geom_boxplot()+
  labs(x="Continent", y="Life Exp", title="Dist of Life Expectancies by Continent & Year")

##################
##Two-Way tables##
##################

##create new variable expectancy
Data<-Data%>%
  mutate(expectancy=ifelse(lifeExp<70,"Low","High"))

##two way table of continent and expectancy
mytab2<-table(Data$continent, Data$expectancy)
##continent in rows, expectancy in columns
mytab2

##convert to proportions based on explanatory variable
##in this example, continent is the explanatory variable

prop.table(mytab2, 1) 
##2nd argument is 1 as we want row props to add to 1

##convert to percent and round to 2dp
round(prop.table(mytab2, 1) * 100, 2)

#######################
##Bar chart (stacked)##
#######################
##useful for two categorical variables



ggplot(Data, aes(x=continent, fill=expectancy))+
  geom_bar(position = "stack")+
  labs(x="Continent", y="Count", title="Life Expectancies by Continent")

ggplot(Data, aes(x=continent, fill=expectancy))+
  geom_bar(position = "dodge")+
  labs(x="Continent", y="Count", title="Life Expectancies by Continent") 

##better for proportions
ggplot(Data, aes(x=continent, fill=expectancy))+
  geom_bar(position = "fill")+
  labs(x="Continent", y="Proportion", title="Proportion of Life Expectancies by Continent")

################
##Scatterplots##
################
##useful for two quantitative variables

##Life expectancy against GDP
ggplot(Data, aes(x=gdpPercap,y=lifeExp))+
  geom_point()

##there may be points which overlap each other, 
##i.e. values that are repeated
ggplot(Data, aes(x=gdpPercap,y=lifeExp))+
  geom_point(alpha=0.2)+
  labs(x="GDP", y="Life Exp", title="Scatterplot of Life Exp against GDP")
