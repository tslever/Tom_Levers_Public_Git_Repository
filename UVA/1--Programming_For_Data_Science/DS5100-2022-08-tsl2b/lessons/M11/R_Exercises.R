# 11.1: Install tidyverse.
#library(dplyr)
library(tidyverse)

# 11.2: Create a tibble to use in these exercises. You can find a text file called exercise-data.txt in the directory of today's module (M11).
scores <- tibble(
    name = c("mike", "carol", "greg", "marcia", "peter", "jan", "bobby", "cindy", "alice"),
    school = c("south", "south", "south", "south", "north", "north", "north", "south", "south"),
    teacher = c("johnson", "johnson", "johnson", "johnson",  "smith", "smith", "smith", "perry", "perry"),
    sex = c("male", "female", "male", "female", "male", "female", "male", "female", "female"),
    math_score = c(4, 3, 2, 4, 3, 4, 5, 4, 5),
    reading_score = c(1, 5, 2, 4, 5, 4, 1, 5, 4)
)

# 11.3: View the tibble.
print(scores)

# 11.4: Make sure you understand the difference between doing something and assigning the result to a new name and just doing the thing.
# First, just get the first three rows of the tibble. Then, assign the first three rows to a variable.
print(scores %>% slice(1:3))
the_slice <- scores %>% slice(1:3)
print(the_slice)

# 11.5: Sort the data by math_score from high to low. Bobby and Alice have the highest math score of 5.
print(scores %>% arrange(desc(math_score)))

# 11.6: Sort the data by name from first to last in the alphabet.
print(scores %>% arrange(name))

# 11.7: Sort the data by sex so females show up first.
print(scores %>% arrange(sex))

# 11.8: Sort the data by school, teacher, sex, math_score, and reading_score
print(scores %>% arrange(school, teacher, sex, math_score, reading_score))

# 11.9: Select on the name, math_score, and reading_score columns.
print(scores %>% select(name, math_score, reading_score))
 
# 11.10: Select all of the columns except the sex column.
print(scores %>% select(-sex))

# 11.11: Select all of the columns except the math_score and reading_score columns.
print(scores %>% select(-math_score, -reading_score))

# 11.12: Keep all of the columns but rearrange them so sex is the first column.
print(scores %>% select(sex, everything())) # Actually, everything else

# 11.13: Filter to students who are male and went to south.
print(scores %>% filter(sex == 'male' & school == 'south'))

# 11.14: Filter to students who did above average in math.
print(scores %>% filter(math_score > mean(math_score)))

# 11.15: Use filter to figure out how many students had a math score of 4 or more and a reading score of 3 or more.
print(scores %>% filter(math_score >= 4 & reading_score >= 3))

# 11.16: Filter to students who got a 3 or worse in either math or reading.
print(scores %>% filter(math_score <= 3 | math_score <= 3))

# 11.17: Filter to students who got a reading score of 2, 3, or 4.
print(scores %>% filter(reading_score == 2 | reading_score == 3 | reading_score == 4))
print(scores %>% filter(reading_score %in% 2:4))

# 11.18: Filter to students who have a name that starts with an 'm'.
# Hint: Type '?substr' in the console and then scroll to the bottom of the help file to see useful examples.
scores %>% filter(substr(name, 1, 1) == 'm')

# 11.19: Filter to teachers whose best math student got a score of 5.
scores_grouped_by_teacher <- scores %>% group_by(teacher)
print(scores_grouped_by_teacher)
print(scores_grouped_by_teacher %>% filter(max(math_score) == 5))
print(scores %>% filter(math_score == 5))
print(scores %>% group_by(teacher) %>% summarize(max_math = max(math_score)) %>% filter(max_math == 5) %>% select(teacher))

# 11.33: Find the maximum math score for each teacher
print(scores %>% group_by(teacher) %>% summarize(max_math_score = max(math_score)))