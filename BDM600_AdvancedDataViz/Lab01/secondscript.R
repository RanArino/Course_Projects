# Generate three random normal vectors of length 100
x1 <- rnorm(100)
x2 <- rnorm(100)
x3 <- rnorm(100)

# Create a data frame t with three columns
t <- data.frame(a = x1, b = x1 + x2, c = x1 + x2 + x3)

# Plot the data frame
plot(t)
# Calculate the standard deviation of each column
print(sd(t$a))
print(sd(t$b))
print(sd(t$c))