# lab05
# math
math <- matrix(c(41, 10, 32, 2), nrow = 2, byrow = TRUE)
dimnames(math) <- list(grades = c("F-B", "A"), study_habit = c("studied", "no_studying"))

print(math)

result <- chisq.test(math, correct = FALSE)
print(result)


# Eng
eng <- matrix(c(4, 12, 11, 31), nrow = 2, byrow = TRUE)
dimnames(eng) <- list(grades = c("F-B", "A"), study_habit = c("studied", "no_studying"))

print(eng)

result <- chisq.test(eng, correct = FALSE)
print(result)


# science
sci <- matrix(c(5, 10, 15, 30), nrow = 2, byrow = TRUE)
dimnames(sci) <- list(grades = c("F-B", "A"), study_habit = c("studied", "no_studying"))

print(sci)

result <- chisq.test(sci, correct = FALSE)
print(result)


# marginal ignore subjects
subject <- matrix(c(50, 32, 58, 63), nrow = 2, byrow = TRUE)
dimnames(subject) <- list(grades = c("F-B", "A"), study_habit = c("studied", "no_studying"))

print(subject)

result <- chisq.test(subject, correct = FALSE)
print(result)
