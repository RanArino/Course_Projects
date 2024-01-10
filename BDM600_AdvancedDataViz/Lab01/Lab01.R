# Q1: Calculate percentage of university life
print((2024 - 2021) / (2024 - 2000) * 100)

# Q2: Assign numbers to variables
start_year <- 2021  # university start year
birth_year <- 2000  # birth year
current_year <- 2024  # current year
percentage <- ((current_year - start_year) / (current_year - birth_year)) * 100
print(percentage)

# Q3: Scalers and sum
vec <- c(4, 5, 8, 11)
print(sum(vec))

# Q4: Plots
rand_num <- rnorm(100)
plot(rand_num)

# Q5: Help function
help(sqrt)

# Q6: Script
source("firstscript.R")

# Q7: Matrix
p <- seq(31, 60)
print(p)
q <- matrix(p, nrow = 6, ncol = 5)
print(q)

# Q8: Data Frame
#  sd: calculate the standard deviation
source("secondscript.R")

# Q9: Graphics
#  rgb: assign RGB color
#  lwd: line width
#  pch: plotting point character
#  cex: size of the objects
source("thirdscript.R")

# Q10: read data frame
data <- read.table("tst1.txt", header = TRUE)
data$g <- data$g * 5
write.table(data, file = "tst2.txt", row.names = FALSE)

# Q11: Table
#  suppose that the tst1.txt has already defined
path <- "Lab01/tst1.txt"
data <- read.table(file = path, header = TRUE)
data$g <- data$g * 5
write.table(data, file = "tst2.txt", row.name = FALSE)

# Q12: Not Avaiable Data
#  return NaN
rand_num <- rnorm(100)
mean_squart <- mean(sqrt(rand_num))
print(mean_squart)

# Q13: Date
dates <- as.Date(c("2024-01-09", "2014-12-05", "2024-12-04"))
presents <- c(3, 1, 2)
plot(dates, presents, type = "o", col = "blue", xlab = "Date", 
     ylab = "Number of Presents")

# Q14: For-loop operation
vec <- seq(1, 100)
for (i in seq_along(vec)) {
  if (vec[i] < 5 || vec[i] > 90) {
    vec[i] <- vec[i] * 10
  } else {
    vec[i] <- vec[i] * 0.1
  }
}

# Q15: Function
func <- function(vec) {
  result <- numeric(length(vec))
  for (i in seq_along(vec)) {
    if (vec[i] < 5 || vec[i] > 90) {
      result[i] <- vec[i] * 10
    } else {
      result[i] <- vec[i] * 0.1
    }
  }
  return(result)
}
vec <- seq(1, 30)
func(vec)
