######################################
##Data Visualization 4: Multivariate##
######################################

library(tidyverse)
library(gapminder)

##we want to focus on most recent data, from year 2007 and 
##add a binary variable for expectancy
Data<-gapminder%>%
  mutate(expectancy=ifelse(lifeExp<70,"Low","High"))%>%
  filter(year==2007)

##another data frame across all years plus a binary variable 
##for expectancy
Data.all<-gapminder%>%
  mutate(expectancy=ifelse(lifeExp<70,"Low","High"))

#############
##Barcharts##
#############

##Barchart for two categorical variables
ggplot(Data,aes(x=continent, fill=expectancy))+
  geom_bar(position = "fill")

##Barchart for two categorical variables across years
##i.e. three categorical variables
ggplot(Data.all,aes(x=continent, fill=expectancy))+
  geom_bar(position = "fill")+
  facet_wrap(~year)

##nicer labels
ggplot(Data.all,aes(x=continent, fill=expectancy))+
  geom_bar(position = "fill")+
  facet_wrap(~year)+
  theme(axis.text.x = element_text(angle = 90))

################
##Scatterplots##
################

##Life expectancy against GDP
ggplot(Data, aes(x=gdpPercap,y=lifeExp))+
  geom_point(alpha=0.2)+
  labs(x="GDP", y="Life Exp", title="Scatterplot of Life Exp against GDP")

##Life expectancy against GDP, with population size. 
##So 3 quantitative variables
ggplot(Data, aes(x=gdpPercap, y=lifeExp, size=pop))+
  geom_point()

##change the size of the plots
ggplot(Data, aes(x=gdpPercap, y=lifeExp, size=pop))+
  geom_point()+
  scale_size(range = c(0.1,12))

##Life expectancy against GDP, with population size and also continent
##So 3 quantitative variables and 1 categorical variable
ggplot(Data, aes(x=gdpPercap, y=lifeExp, size=pop, color=continent))+
  geom_point()+
  scale_size(range = c(0.1,12))

##change plot shape and make plots a bit more translucent for overlaps
ggplot(Data, aes(x=gdpPercap, y=lifeExp, size=pop, fill=continent))+
  geom_point(shape=21, alpha=0.5)+
  scale_size(range = c(0.1,12))

