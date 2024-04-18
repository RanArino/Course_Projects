# import all libraries
library(ca)
library(car) 
library(dplyr)
library(forcats)
library(ggplot2)
library(MASS)
library(vcdExtra)

# import data
data("Accident", package="vcdExtra")
help(Accident)
tail(Accident)
print(dim(Accident))

# check dimension
dim(Accident)

# check fundamental information
summary(Accident)

# Bar plot: Number of people with different modes in accident
Accident %>%
  group_by(mode) %>%
  summarise(freq = sum(Freq)) %>%
  ggplot(aes(x=fct_reorder(mode,freq,.desc = TRUE),y=freq)) +
  geom_bar(stat = "identity",fill = "cornflowerblue", width = 0.5) +
  ggtitle("Number of people with different modes in accident") +
  xlab("") +
  theme(panel.grid.major.x = element_blank())

# Bar plot: Number of people of different ages in accident
Accident %>%
  group_by(age) %>%
  summarise(freq = sum(Freq)) %>%
  ggplot(aes(x=age,y=freq)) +
  geom_bar(stat = "identity",fill = "cornflowerblue",  width = 0.7) +
  ggtitle("Number of people of different ages in accident") +
  xlab("") +
  coord_flip() +
  theme(panel.grid.major.x = element_blank())

# Use loglm() to fit the model of mutual independence
accident_model <- loglm(Freq ~ age + mode + gender + result, data = Accident)
accident_model

# Mosaic plot
mosaic(accident_model, labeling_args = list(abbreviate = TRUE))

# Multiple correspondence analysis
#  Convert data frame to a 4-way table
accident_tab <- xtabs(Freq ~ age + mode + gender + result, data = Accident)
structable(mode + gender ~ age + result, data=accident_tab)
summary(accident_tab)
#  Correspondence analysis
accident_mca <- mjca(accident_tab)
summary(accident_mca)
#  MCA plot
res <- plot(accident_mca, labels = 0, pch = ".", cex.lab = 1.0)
#  Extract factor names and levels
coords <- data.frame(res$cols, accident_mca$factors)
nlev <- accident_mca$levels.n
fact <- unique(as.character(coords$factor))
cols <- c("blue", "red", "green", "black")
points(coords[,1:2], pch=rep(16:19, nlev), col=rep(cols, nlev), cex=1.0)
text(coords[,1:2], label=coords$level, col=rep(cols, nlev), pos=2,
cex=1.0, xpd=TRUE)
lwd <- c(2, 2, 2, 4)
for(i in seq_along(fact)) {
lines(Dim2 ~ Dim1, data = coords, subset = factor==fact[i],
lwd = lwd[i], col = cols[i])
}
legend("bottomright",
legend = c("Age", "Mode", "Gender", "Result"),
title = "Factor", title.col = "black",
col = cols, text.col = cols, pch = 12:15,
bg = "gray95", cex = 0.8)

# Construct 2D plot
plot(accident_mca)
mcaplot(accident_mca)


# Exploratory analysis
#  Mutual independence
acc.mod0 <- glm(Freq ~ age + result + mode + gender, 
                data=Accident, 
                family=poisson)
LRstats(acc.mod0)

mosaic(acc.mod0, ~mode + age + gender + result)

# result as a response
acc.mod1 <- glm(Freq ~ age*mode*gender + result, 
                data=Accident, 
                family=poisson)
LRstats(acc.mod1)

mosaic(acc.mod1, ~mode + age + gender + result, 
    labeling_args = list(abbreviate = c(gender=1, result=4)))

# allow two-way association of result with each explanatory variable
acc.mod2 <- glm(Freq ~ age*mode*gender + result*(age+mode+gender), 
                data=Accident, 
                family=poisson)
LRstats(acc.mod2)
mosaic(acc.mod2, ~mode + age + gender + result, 
    labeling_args = list(abbreviate = c(gender=1, result=4)))

acc.mods <- glmlist(acc.mod0, acc.mod1, acc.mod2)
LRstats(acc.mods)

## Binomial (logistic regression) models for result
## ------------------------------------------------
acc.bin1 <- glm(result=='Died' ~ age + mode + gender, 
    weights=Freq, data=Accident, family=binomial)
Anova(acc.bin1)

acc.bin2 <- glm(result=='Died' ~ (age + mode + gender)^2, 
    weights=Freq, data=Accident, family=binomial)
Anova(acc.bin2)

acc.bin3 <- glm(result=='Died' ~ (age + mode + gender)^3, 
    weights=Freq, data=Accident, family=binomial)
Anova(acc.bin3)

# compare models
anova(acc.bin1, acc.bin2, acc.bin3, test="Chisq")
