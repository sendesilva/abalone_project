
### The purpose of this project is to apply machine learning techniques such as linear and 
# polynomial regression to the abalone dataset, from the UCI repository, to enable predicting
# the age (number of shell rings) of an abalone from its physical charateristics. 

# The dataset provides 8 physical characteristics (sex, length, diameter, height, whole weight,
# shucked weight, viscera weight, shell weight) as possible predictors in addition to the the 
# number of shell rings (henceforth referred to as rings for convenience). This equates to a 
# total of 9 attributes. The number of rings (integer), which varies from 1 to 29, +1.5 is 
# thought to be a reasonable estimate of the age in years. There are a total of 4177 samples or 
# records. For this project, the rings outcome variable is treatd as a continuous variable as it 
# ranges from 1 to 29 with an intervals of 1.

## Dataset selected: +Abalone+ from the UCI Machine Learning Repository 
## Repository citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
## [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information 
## and Computer Science.
## Dataset source: Abalone Data Set from UCI Machine Learning Repository,
## https://archive.ics.uci.edu/ml/datasets/Abalone

## Additional references:
## 1. Kevin Markham https://github.com/justmarkham work on Abalone Dataset at GitHub
## https://github.com/ajschumacher/gadsdc1/blob/master/dataset_research/kevin_abalone_dataset.md

# Predicting the age of abalone from physical measurements. The age of abalone is determined by 
# cutting the shell through the cone, staining it, and counting the number of rings through a 
# microscope -- Rings	(integer)	+1.5 is thought to be a reasonable estimate of the age in years.
# Other measurements, which are easier to obtain, are used to predict the age. 

# 'rings' is treated as a continuous variable where required which should not be a problem as
# as there are 29 values with any rounding affecting actual value by +/-0.5 at most which is
# about 1.5%.

# Install packages used, if required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")

# load libraries
library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(dplyr)
library(ggplot2)
library(GGally)

# Abalone dataset:
# https://archive.ics.uci.edu/ml/datasets/Abalone

### Auto-download and read the dataset into a data frame
abalone <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header=FALSE)

# add column names
names(abalone) <- c("sex", "length", "diameter", "height", "weight.whole",
                    "weight.shucked", "weight.viscera", "weight.shell", "rings")

# confirm that there are 4177 observations
nrow(abalone)

# confirm that there are no missing values
sum(is.na(abalone))

# Abalone Data explore
# data(abalone) # no dataset - now in dataframe
names(abalone)
dim(abalone)
head(abalone)
str(abalone)
summary(abalone)
head(abalone$rings)
table(abalone$sex)


### Create train set and validation set for testing train set predictions
# Validation set will be 10% of abalone data derived from test_index
set.seed(1, sample.kind="Rounding")

# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = abalone$rings, times = 1, p = 0.1, list = FALSE)
train <- abalone[-test_index,]
validation <- abalone[test_index,]

# remove unecessary data
rm(test_index, abalone)

### Methods and Analysis
## Data Cleaning / Preparation
# From the original data examples with missing values were removed from the abalone dataset 
# (the majority having the predicted value missing), and the ranges of the continuous values
# have been scaled (by dividing by 200) for use with an ANN (abalone dataset readme file 
# - abalone.names).

## Visualization / Exploration of train dataset
head(train)
str(train)
# number of unique rings and distribution
train %>% summarise(n_rings = n_distinct(rings))
table(train$rings)

# visualizing correlation between rings and external characteristics of abalone
pairs(~ rings + weight.whole + diameter + length + height, data = train)
cor(train[, c(9,2,3,4,5)])

# adding correlation values using ggpairs
ggpairs(data = train, columns = c(9, 2,3,4,5))  # scatterplots dont seem to match corr 
# values but agree with cor values

# check whole dataset
ggpairs(train)

## Check on validation set
head(validation)
str(validation)
# number of unique rings
validation %>% summarise(n_rings = n_distinct(rings))


## Determining accuracy of prediction system
# To compare different models root mean square estimate (RMSE) is used as the loss function 
# on the test set.

# RMSE value
RMSE <- function(true_rings, predicted_rings){
  sqrt(mean((true_rings - predicted_rings)^2)) }

### Building Prediction System

## Model 1: reference or just the average
# This will represent the worst possible prediction to be modelled.Assumes all abalone 
# regardless of the physical characteristics have the same num of rings
# with the differences explained by random variation.

mu_hat <- mean(train$rings)
mu_hat

# checking RMSE
reference_rmse <- RMSE(validation$rings, mu_hat)
reference_rmse

# Creating a table to store and compare accuracies of different models
rmse_results <- data_frame(method = "Model 1: Just the average", RMSE = reference_rmse)
rmse_results %>% knitr::kable()

## Model 2: LR - External physical characteristics

# Linear regression using the lm function and with external physical characteristics as 
# predictors of num of rings as outcome This does not require the physical charateristics post
# cutting open an abalone. Hence this method could help abalone farmers, sellers and buyers to
# value an abalone without having to cut it open.


fit2 <- lm(rings ~ weight.whole + diameter + length + height, data = train)
fit2
summary(fit2)

# using model, fit, to predict rings in validation set
p2 <- predict(fit2, newdata = validation)

str(p2)
class(p2)

# accuracy
rmse_model_2 <- RMSE(validation$rings, p2)
rmse_model_2

rmse_results <- bind_rows(rmse_results, data_frame(method="Model 2: LR - External physical 
                                                   characteristics", RMSE = rmse_model_2 ))
rmse_results %>% knitr::kable()


## Model 3: LR - External physical characteristics less weight.whole 
# Model 2 is better than the average but still has a high RMSE. However it was observed in the
# summary of the fitted model that the p value for the weight.whole variable was very high 
# unlike the other variables and intercept. 
# Hence model 3 will use model 2 without this variable looking for a better RMSE

fit3 <- lm(rings ~ diameter + length + height, data = train)
fit3
summary(fit3)

# using model, fit, to predict rings in validation set
p3 <- predict(fit3, newdata = validation)

# accuracy
rmse_model_3 <- RMSE(validation$rings, p3)
rmse_model_3
# comment: does not help to remove weight.whole as the RMSE became higher i.e. less accurate

rmse_results <- bind_rows(rmse_results, data_frame(method="Model 3: LR - External pc less 
                                                   weight.whole", RMSE = rmse_model_3 ))
rmse_results %>% knitr::kable()


## Model 4: LR - Using all predictors
# Models 2 and 3 still have a high RMSEs hence the need to explore the use of all 8 attributes 
# as predictors.

fit4 <- lm(rings ~ sex + length + diameter + height + weight.whole + weight.shucked
           + weight.viscera + weight.shell, data = train)
fit4
summary(fit4)

# using model, fit4, to predict rings in validation set
p4 <- predict(fit4, newdata = validation)

# accuracy
rmse_model_4 <- RMSE(validation$rings, p4)
rmse_model_4

rmse_results <- bind_rows(rmse_results, data_frame(method="Model 4: LR - using all predictors",
                                                   RMSE = rmse_model_4 ))
rmse_results %>% knitr::kable()


## Model 5: LR - Using all predictors less length
# Models 4 summary statistics show that length is not significant at p = 0.05 and hence this 
# model runs model 4 without lenght as a predictor.

fit5 <- lm(rings ~ sex + diameter + height + weight.whole + weight.shucked
           + weight.viscera + weight.shell, data = train)
fit5
summary(fit5)

# using model, fit, to predict rings in validation set
p5 <- predict(fit5, newdata = validation)

# accuracy
rmse_model_5 <- RMSE(validation$rings, p5)
rmse_model_5 

rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Model 5: LR - using all precdictors less length", 
                                                   RMSE = rmse_model_5 ))
rmse_results %>% knitr::kable()
# produces a very slight improvement in RMSE


## Model 6 Testing use of polynomial regression
#  TEST1 rings vs weight.whole poly(3) 
poly3model <- lm(rings ~ poly(weight.whole, 3), data = train)
poly3model
summary(poly3model)

# using model, poly3model, to predict rings in validation set
p3m_ww <- predict(poly3model, newdata = validation)

# accuracy
rmse_p3m_ww <- RMSE(validation$rings, p3m_ww)
rmse_p3m_ww 


#  TEST2 rings vs weight.whole poly(2) 
poly2model <- lm(rings ~ poly(weight.whole, 2), data = train)
poly2model

summary(poly2model)

# using model, poly2model, to predict rings in validation set
p2m_ww <- predict(poly2model, newdata = validation)

# accuracy
rmse_p2m_ww <- RMSE(validation$rings, p2m_ww)
rmse_p2m_ww 

#  TEST3 rings vs diameter poly(2) 
poly2model_dia <- lm(rings ~ poly(diameter, 2), data = train)
poly2model_dia

summary(poly2model_dia)

# using model, poly3model_dia, to predict rings in validation set
p2m_d <- predict(poly2model_dia, newdata = validation)

# accuracy
rmse_p2m_d <- RMSE(validation$rings, p2m_d)
rmse_p2m_d 

#  TEST4 rings vs all predictors excluding sex
poly2model_all_pc <- lm(rings ~ sex + poly(diameter,2) + poly(height,2) + 
                           poly(weight.whole,2) + poly(weight.shucked,2) + 
                           poly(weight.viscera,2) + poly(weight.shell, 2), data = train)
poly2model_all_pc
summary(poly2model_all_pc)

# using model, poly2model_all_pc, to predict rings in validation set
p2m_all_pc <- predict(poly2model_all_pc, newdata = validation)

# accuracy
rmse_p2m_all_pc <- RMSE(validation$rings, p2m_all_pc)
rmse_p2m_all_pc 

# Choose TEST4 as model 6 due best accuracy
# Comparing with other results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 6: PolyR - using all predictors excl sex", 
                                                   RMSE = rmse_p2m_all_pc ))
rmse_results %>% knitr::kable()
