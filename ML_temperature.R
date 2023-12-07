# In this file we fit machine learning models to the temperature variables.

library(xgboost)
library(caret)
library(ggplot2)
library(gbm)
library(dplyr)
library(knitr) # table format in output
library(lubridate)
library(tidyr)
library(cowplot) 
library(xts)
library(TSA)
library(tseries)
library(caschrono)
library(forecast)
rm(list = ls())
source("importation_datasets.R")


# We're doing some preprocessing on the data to select only the important variables
data_temp <- rbind(data_temp1, data_temp2, data_temp3, data_temp4)
data_temp <- data_temp %>% arrange(date)
data_temp$date <- as.Date(data_temp$date, format="%Y-%m-%d")
data_temp <- data_temp %>% select(-X)
data_temp <- data_temp %>% select(-year)
data_temp <- data_temp %>%
  mutate(year = lubridate::year(date),
         month = lubridate::month(date, label = FALSE))

# Here we transform daily data_temp to monthly data_temp
data_temp <- data_temp %>%
  group_by(year, month) %>%
  summarise(temperature = mean(moyenne_temperature_celsius, na.rm = TRUE))
data_temp$date <- paste(data_temp$year,data_temp$month, sep = "-")
data_temp$date <- as.Date(paste0(data_temp$date, "-1"), format="%Y-%m-%d")


# We want to create lag for the previous day, the mean of the last 7 and 30 days and the same day one year ago
data_temp$lag1 <- lag(data_temp$temperature, 1)
data_temp$lag2 <- lag(data_temp$temperature, 2)
data_temp$lag3 <- lag(data_temp$temperature, 3)
data_temp$lag4 <- lag(data_temp$temperature, 4)
data_temp$lag5 <- lag(data_temp$temperature, 5)
data_temp$lag6 <- lag(data_temp$temperature, 6)
data_temp$lag12<- lag(data_temp$temperature, 12)

# We want to retain only the non NA part ie we lose a year
data_temp <- data_temp[13:490,]

# Splitting the data into test and train samples
train_data <- subset(data_temp, year >= 1984 & year <= 2015)
test_data <- subset(data_temp, year >= 2016 & year <= 2022)
test_data <- test_data[,-1]
train_data <- train_data[,-1]

# We tranform the variable Months to encode it later for xgboost
train_data$month <- as.factor(train_data$month)
test_data$month <- as.factor(test_data$month)


# We scale and reduce both samples. We scale with the mean and variance of the training dataset
# to avoid tacking information from the testing sample. 
# We also normalize all the variables
y_train_mean <- mean(train_data$temperature)
y_train_sd <- sd(train_data$temperature)
train_data$temperature<- (train_data$temperature-y_train_mean)/y_train_sd
test_data$temperature<- (test_data$temperature-y_train_mean)/y_train_sd

lag1_train_mean <- mean(train_data$lag1)
lag1_train_sd <- sd(train_data$lag1)
train_data$lag1 <- (train_data$lag1-lag1_train_mean)/lag1_train_sd
test_data$lag1 <- (test_data$lag1-lag1_train_mean)/lag1_train_sd

lag6_train_mean <- mean(train_data$lag6)
lag6_train_sd <- sd(train_data$lag6)
train_data$lag6 <- (train_data$lag6-lag6_train_mean)/lag6_train_sd
test_data$lag6 <- (test_data$lag6-lag6_train_mean)/lag6_train_sd

lag2_train_mean <- mean(train_data$lag2)
lag2_train_sd <- sd(train_data$lag2)
train_data$lag2 <- (train_data$lag2-lag2_train_mean)/lag2_train_sd
test_data$lag2 <- (test_data$lag2-lag2_train_mean)/lag2_train_sd

lag3_train_mean <- mean(train_data$lag3)
lag3_train_sd <- sd(train_data$lag3)
train_data$lag3 <- (train_data$lag3-lag3_train_mean)/lag3_train_sd
test_data$lag3 <- (test_data$lag3-lag3_train_mean)/lag3_train_sd

lag4_train_mean <- mean(train_data$lag4)
lag4_train_sd <- sd(train_data$lag4)
train_data$lag4 <- (train_data$lag4-lag4_train_mean)/lag4_train_sd
test_data$lag4 <- (test_data$lag4-lag4_train_mean)/lag4_train_sd

lag5_train_mean <- mean(train_data$lag5)
lag5_train_sd <- sd(train_data$lag5)
train_data$lag5 <- (train_data$lag5-lag5_train_mean)/lag5_train_sd
test_data$lag5 <- (test_data$lag5-lag5_train_mean)/lag5_train_sd

lag12_train_mean <- mean(train_data$lag12)
lag12_train_sd <- sd(train_data$lag12)
train_data$lag12 <- (train_data$lag12-lag12_train_mean)/lag12_train_sd
test_data$lag12 <- (test_data$lag12-lag12_train_mean)/lag12_train_sd


# Now we encode the month variable
categorical_vars <- sapply(train_data, is.factor)
train_encoded <- cbind(train_data[, !categorical_vars, drop = FALSE], model.matrix(~ . - 1, train_data[, categorical_vars]))
test_encoded <- cbind(test_data[, !categorical_vars, drop = FALSE], model.matrix(~ . - 1, test_data[, categorical_vars]))
train_encoded <- train_encoded %>% select(-date)
test_encoded <- test_encoded %>% select(-date)

# Setup train data to be understood by xgboost
X_train <- train_encoded[, -1, drop = FALSE]
y_train <- train_encoded[, 1]
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)

# Setup data to be understood by xgboost
X_test <- test_encoded[, -1, drop = FALSE]
y_test <- test_encoded[, 1]
dtest <- xgb.DMatrix(data = as.matrix(X_test))


# Setup cross validation parameters
ctrl <- trainControl(method = "cv", number = 5)    # You can adjust the number of folds

# Set the parameters to tune for xgBoost
param_grid <- data.frame(
  nrounds = c(100, 500, 1000),                 # Adjust these values based on your preferences
  max_depth = c(6, 9, 15),                     # Max Tree Depth
  eta = c(0.01, 0.1, 0.3),                    # Shrinkage
  gamma = c(0, 1, 2),                         # Minimum Loss Reduction
  subsample = c(0.7, 0.8, 0.9),               # Subsample Percentage
  colsample_bytree = c(0.7, 0.8, 0.9),       # Subsample Ratio of Columns
  rate_drop = c(0.1, 0.2, 0.3),               # Fraction of Trees Dropped
  skip_drop = c(0.1, 0.2, 0.3),               # Prob. of Skipping Drop-out
  min_child_weight = c(1, 3, 5)               # Minimum Sum of Instance Weight
)

# We write here the formula for regression
xgb_formula <- as.formula("y_train ~ .")
# We create here a data frame for the cross validation
train_data_xgb <- data.frame(y_train, X_train)

#We do the cross validation to find the best parameter
xgboostCrossValidation <- train(
  xgb_formula,
  data = train_data_xgb,
  method = "xgbDART",
  trControl = ctrl,
  tuneGrid = param_grid
)
#We get the best tune
xgboostCrossValidation$bestTune

# Parameters for xgboost
params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 9,
  gamma = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight =3,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

# We train the model
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 500, verbose = 0)

# Feature importance
importance_matrix <- xgb.importance(
  feature_names = colnames(dtrain),
  model = xgb_model
)
xgb.plot.importance(importance_matrix)

# Make predictions on the test data
y_pred_xg <- predict(xgb_model, newdata = dtest)
MSE_xgboost <- sum((y_test-y_pred_xg)^2)

# Plot the test_data and the predictions
ggplot() +   
  geom_line(aes(x = test_data$date, y = y_test, colour = "Test sample"), size = 0.8) + 
  geom_line(aes(x = test_data$date, y = y_pred_xg, colour = "Predictions"), size = 1) +   
  theme_minimal() +   
  labs(y = "Temperature", x = "Date") +   
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red")) 


##GBM, We do the same that for xgboost
train_data_gbm <- data.frame(y_train, X_train)
gbm_formula <- as.formula("y_train ~ .")

param_grid <- data.frame(n.trees= c(10, 50, 100, 1000, 10000),
                         interaction.depth= c(1, 3, 5, 10, 20),
                         shrinkage= c(0.1, 0.01, 0.001, 0.0001, 0.00001),
                         n.minobsinnode = c(3, 5, 10, 20, 2))


gbmCrossValidation <- train(gbm_formula, 
                            data = train_data_gbm,
                            method = "gbm",
                            distribution = "gaussian",
                            trControl = ctrl,   
                            tuneGrid = param_grid) 

gbmCrossValidation$bestTune 

gbm_model <- gbm(formula = gbm_formula, 
                 data = train_data_gbm, 
                 distribution = "gaussian", 
                 shrinkage=0.01,
                 n.trees=1000,
                 interaction.depth = 10,
                 n.minobsinnode = 1)


y_pred_gbm <- predict(gbm_model, newdata=X_test)
MSE_gbm <- sum((y_test-y_pred_gbm)^2)

ggplot() +   
  geom_line(aes(x = test_data$date, y = y_test, colour = "Test sample"), size = 0.8) + 
  geom_line(aes(x = test_data$date, y = y_pred_gbm, colour = "Predictions"), size = 1) +   
  theme_minimal() +   
  labs(y = "Temperature", x = "Date") +   
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))


