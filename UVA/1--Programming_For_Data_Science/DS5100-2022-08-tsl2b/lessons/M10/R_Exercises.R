# 10.1: Write an R program to get the first 10 Fibonacci numbers.

vector_of_Fibonacci_numbers <- numeric(10)
vector_of_Fibonacci_numbers[1] <- 1
vector_of_Fibonacci_numbers[2] <- 2
for (i in 3:10) {
    vector_of_Fibonacci_numbers[i] <- vector_of_Fibonacci_numbers[i - 2] + vector_of_Fibonacci_numbers[i - 1]
}
print(vector_of_Fibonacci_numbers)


# 10.2: Convert the previous code into a function. Instead of printing the results, just return the vector.

get_vector_of_Fibonacci_numbers <- function() {
    vector_of_Fibonacci_numbers <- numeric(10)
    vector_of_Fibonacci_numbers[1] <- 1
    vector_of_Fibonacci_numbers[2] <- 2
    for (i in 3:10) {
     vector_of_Fibonacci_numbers[i] <- vector_of_Fibonacci_numbers[i - 2] + vector_of_Fibonacci_numbers[i - 1]
    }
    return(vector_of_Fibonacci_numbers)
}
print(get_vector_of_Fibonacci_numbers())


# 10.3: Write a function that takes a numeric x and return 1 if 0 <= x <= 1. Return 0 otherwise.

is_between_zero_and_one_inclusive <- function(x) {
    if ((0 <= x) & (x <= 1)) {
        return(1)
    } else {
        return(0)
    }
}


# 10.4: Test the function on some values

print(is_between_zero_and_one_inclusive(-0.1))
print(is_between_zero_and_one_inclusive(0))
print(is_between_zero_and_one_inclusive(1))
print(is_between_zero_and_one_inclusive(1.1))


# 10.5: Write an R program to extract the first 10 English letters in lowercase, the last 10 letters in uppercase, and letters between the twenty-second and twenty-fourth letters inclusive in uppercase.
# Hint: R comes with letters and LETTERS, two vectors that contain lowercase and uppercase letters, respectively.

print("First 10 letters in lowercase:")
print(head(letters, 10))
print("Last 10 letters in uppercase:")
print(tail(LETTERS, 10))
print("Letters between twenty-second and twenty-fourth letters:")
print(LETTERS[22:24])

# 10.6: Write an R program to print the numbers from 1 to 100, print "Fizz" for multiples of 3, print "Buzz" for multiples of 5, and print "FizzBuzz" for multiples of both.

for (n in 1:100) {
    a <- ''
    b <- ''
    if (n %% 3 == 0) {
        a <- "Fizz"
    }
    if (n %% 5 == 0) {
        b <- "Buzz"
    }
    print(paste(n, ' ', a, b, sep = ''))
}


# 10.7: Write an R program to get the unique space-separated elements of a given string and unique numbers of a vector.
# Hint: Use the unique function.

sentence <- "The quick brown fox jumps over the lazy dog."
print(sentence)
print(class(sentence))
lowercase_sentence <- tolower(sentence)
print(sentence)
print(class(lowercase_sentence))
vector_of_elements <- strsplit(lowercase_sentence, ' ')[[1]]
print(vector_of_elements)
print(class(vector_of_elements))
vector_of_unique_elements <- unique(vector_of_elements)
print(vector_of_unique_elements)
print(class(vector_of_unique_elements))
vector_of_numbers <- c(1, 2, 2, 3, 4, 4, 5, 6)
print(vector_of_numbers)
print(class(vector_of_numbers))
vector_of_unique_numbers <- unique(vector_of_numbers)
print(vector_of_unique_numbers)
print(class(vector_of_unique_numbers))


# 10.8: Write an R program to find the maximum and minimum value of a given vector.
# Hint: Use the max and min functions.

vector_of_numbers = c(10, 20, 30, 40, 50, 60)
print("Original vector:")
print(vector_of_numbers)
print(class(vector_of_numbers))
print(paste("Maximum value of vector of numbers: ", max(vector_of_numbers), sep = ""))
print(paste("Minimum value of vector of numbers: ", min(vector_of_numbers), sep = ""))


# 10.9: Write an R program to create a sequence of numbers from 20 to 50, find the mean of numbers from 20 to 50, and find the sum of numbers from 51 to 91.

print("Sequence of numbers from 20 to 50:")
print(seq(20, 50))
print("Mean of numbers from 20 to 60:")
print(mean(20:60))
print(mean(seq(20, 60)))
print("Sum of numbers from 51 to 91:")
print(sum(51:91))
print(sum(seq(51, 91)))


# 10.10: Write an R program to create a data frame from four given vectors.

name <- c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
score <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
attempt <- c(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
qualify <- c('Y', 'N', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N', 'Y')
data_frame <- data.frame(name, score, attempt, qualify)
print(data_frame)


# 10.11: Write an R program to extract the third and fifth rows with the first and third columns from the built-in airquality data frame.

print(airquality[c(3, 5), c(1, 3)])


# 10.12: Create a plot to compare gas mileage, number of cylinders, and horsepower using the built-in data frame mtcars.

plot(mtcars[ , c('hp', 'cyl', 'mpg')])