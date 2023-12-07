## In this file you can perform the code for the results in part 4.1 and 4.2 of the report.

rm(list=ls())
library(dplyr)
library(zoo)
library(tidyverse)
library(ggplot2)
library(forecast)
library(tseries)
library(lmtest)
library(xgboost)
library(randomForest)
library(caret)
library(gbm)

# Import of the data we want to study, evaporation
full_data <- read.csv("daily_Sum_Evap.csv")
full_data <- full_data[,-1]
full_data <- full_data[,-3]
full_data$date <- as.Date(full_data$date)

liste_var = c("daily_Sum_Temp2M.csv")

# Adding all variables to the features
for (name_variable in liste_var) {
  print(name_variable)
  data <- read.csv(name_variable)
  
  # Extract the third column and give it a specific name
  col_name <- gsub("daily_Sum_|\\.csv", "", name_variable)  # Extract variable name from file name
  col_values <- data[, 3]
  
  # Assign specific names to the columns
  full_data[[col_name]] <- col_values
}

# Grouping by months
full_data <- full_data %>%
  group_by(date = lubridate::floor_date(date, 'month')) %>%
  summarize(across(c("evaporation_from_open_water_surfaces_excluding_oceans_sum","Temp2M"), sum, na.rm = TRUE))

# Adding year and month
full_data$year <- year(full_data$date)
full_data$month <- month(full_data$date)

# Cerating lag variables for every variables (run only for part 4.1)
for (col in c("evaporation_from_open_water_surfaces_excluding_oceans_sum", "Temp2M")) {
  for (added in c("lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag12")) {
    # Create lag columns
    lag_value <- as.numeric(gsub("lag", "", added))
    full_data <- full_data %>%
      mutate(!!paste0(col, "_", added) := lag(!!sym(col), lag_value))
  }
}

# Splitting the data into test and train samples
train_data <- subset(full_data, year >= 1984 & year <= 2015)
test_data <- subset(full_data, year >= 2016 & year <= 2022)
train_data <- train_data[,-1]
train_data <- train_data[,-3]
data_for_plot <- test_data
test_data <- test_data[,-1]
test_data <- test_data[,-3]

# Set as factor fro machine learning models
train_data$month <- as.factor(train_data$month)
test_data$month <- as.factor(test_data$month)

# Normalization for output variable
y_train_mean <- mean(train_data$evaporation_from_open_water_surfaces_excluding_oceans_sum)
y_train_sd <- sd(train_data$evaporation_from_open_water_surfaces_excluding_oceans_sum)
train_data$evaporation_from_open_water_surfaces_excluding_oceans_sum <- (train_data$evaporation_from_open_water_surfaces_excluding_oceans_sum-y_train_mean)/y_train_sd
test_data$evaporation_from_open_water_surfaces_excluding_oceans_sum <- (test_data$evaporation_from_open_water_surfaces_excluding_oceans_sum-y_train_mean)/y_train_sd
data_for_plot$evaporation_from_open_water_surfaces_excluding_oceans_sum <- (data_for_plot$evaporation_from_open_water_surfaces_excluding_oceans_sum-y_train_mean)/y_train_sd

# Normalization for all variables
for (col in colnames(train_data)) {
  # Exclude "year" and "month" columns from normalization
  if (!(col %in% c("year", "month", "evaporation_from_open_water_surfaces_excluding_oceans_sum"))) {
    # Calculate mean and standard deviation from the training set
    train_mean <- mean(train_data[[col]], na.rm = TRUE)
    train_sd <- sd(train_data[[col]], na.rm = TRUE)
    
    # Normalize the training set
    train_data[[col]] <- (train_data[[col]] - train_mean) / train_sd
    
    # Normalize the test set using the mean and SD from the training set
    test_data[[col]] <- (test_data[[col]] - train_mean) / train_sd
  }
}

# Dropping the temperature (for part 4.1)
columns_to_drop <- c("Temp2M")
train_data <- subset(train_data, select = -which(names(train_data) %in% columns_to_drop))
test_data <- subset(test_data, select = -which(names(test_data) %in% columns_to_drop))

### Run the following lines for part 4.2
# Dropping the temperature in test sample (for part 4.2)
columns_to_drop <- c("Temp2M")
test_data <- subset(test_data, select = -which(names(test_data) %in% columns_to_drop))
# Adding the new predicted temperature from the other sarima model
y_pred_temp_sarima <- read.csv("y_pred_temp_sarima.csv")
test_data <- cbind(test_data, "Temp2M" = y_pred_temp_sarima$x)
# Changes the order of the columns, necessary for exogenous variables
test_data <- test_data[, c("evaporation_from_open_water_surfaces_excluding_oceans_sum", "Temp2M", "month")]


##############
#--SARIMAX--#
#############
get.best.arima <- function(x.ts, maxord = c(1,1,1,1,1,1))
{
  best.aic <- 1e8
  n <- length(x.ts)
  for (p in 0:maxord[1]) for(d in 0:maxord[2]) for(q in 0:maxord[3])
    for (P in 0:maxord[4]) for(D in 0:maxord[5]) for(Q in 0:maxord[6])
    {
      # Fitting the ARIMA model
      fit <- Arima(y = x.ts, 
                   order = c(p,d,q), 
                   seasonal = c(P,D,Q), 
                   method = "CSS",
                   lambda = NULL,
                   include.constant = TRUE,
                   xreg = exogenous_variable)
      
      # Calculates the consistent AIC
      fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
      
      # Saves the model if cAIC is better, ie smaller
      if (fit.aic < best.aic)
      {
        best.aic <- fit.aic
        best.fit <- fit
        best.model <- c(p,d,q,P,D,Q)
      }
    }
  list(best.aic, best.fit, best.model)
}

columns_to_drop_exogenous <- c("evaporation_from_open_water_surfaces_excluding_oceans_sum", "month")
exogenous_variable <- subset(train_data, select = -which(names(train_data) %in% columns_to_drop_exogenous))
exogenous_variable <- as.matrix(exogenous_variable)

# We create a ts object from our data
y <- ts(data = train_data$evaporation_from_open_water_surfaces_excluding_oceans_sum, start = c(1984, 1), end = c(2016, 0), frequency = 12)
plot(y)
bestArima <- get.best.arima(y, maxord = c(2,2,2,2,2,2))

# The model has parameters
bestArima[[3]]

# We retrain the model
myModel <- Arima(y, order = c(0, 0, 0), seasonal = c(0, 1, 1), lambda = NULL, include.constant = TRUE, xreg = exogenous_variable)

# We check the residuals to be white noise
acf(resid(myModel))
pacf(resid(myModel))
autoplot(myModel)
coeftest(myModel)
checkresiduals(myModel)
ggtsdisplay(myModel$residuals)
Box.test(myModel$residuals, lag = 24, fitdf = 1, type = "Ljung") # pvalue > 0.05 means we have white noise


# We check AIC and BIC
myModel$aic
myModel$bic

# Plot of how the model fits the train data
autoplot(y) +
  autolayer(myModel$fitted, series = "ARIMA(0,0,0)(0,1,1)12")

exogenous_variable_test <- subset(test_data, select = -which(names(test_data) %in% columns_to_drop_exogenous))
exogenous_variable_test <- as.matrix(exogenous_variable_test)

# Plot of the predictions of the model
d.forecast <- forecast(myModel, level = c(95), h = 100, xreg = exogenous_variable_test)
autoplot(d.forecast)

# Generate forecasts for the test data
test_predictions <- forecast(myModel, h = nrow(test_data),  xreg = exogenous_variable_test)
data_for_plot$pred <- test_predictions$mean

# Plot the series
ggplot() +
  geom_line(data = data_for_plot, aes(x = date, y = evaporation_from_open_water_surfaces_excluding_oceans_sum, colour = "Test sample"), size = 0.8) +
  geom_line(data = data_for_plot, aes(x = date, y = pred, colour = "Predictions"), size = 1) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))

# Computes the empirical risk
sum((data_for_plot$evaporation_from_open_water_surfaces_excluding_oceans_sum-data_for_plot$pred)^2)










# Specify the target variable
target_variable <- "evaporation_from_open_water_surfaces_excluding_oceans_sum"

# Specify the predictor variables (features)
predictor_variables <- setdiff(names(train_data), target_variable)

# Set up the formula
formula <- as.formula(paste(target_variable, "~", paste(predictor_variables, collapse = "+")))

##################
#-RANDOM FOREST--#
##################

# Cross Validation
param_grid <- data.frame(
  mtry = c(2, 4, 6, 8, 10)
)
ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

randomforestCrossValidation <- train(
  formula,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneGrid = param_grid,
  ntree = 1000
)
randomforestCrossValidation$bestTune

# Training the best Random Forest
randomforestModel <- randomForest(
  formula,
  data = train_data,
  ntree = 1000,            # You can adjust the number of trees
  mtry = 2
)

# Make predictions on the test set
y_pred_rf <- predict(randomforestModel, newdata = test_data)

# Get feature importance
importance_matrix <- importance(randomforestModel)
# Print the importance matrix
print(importance_matrix)
# Plot feature importance
varImpPlot(randomforestModel)

# Assuming you have vectors y_test and y_pred, and the month information
# Create a data frame
plot_data <- cbind(data_for_plot, y_pred_rf)

# Plot the series
ggplot() +
  geom_line(data = plot_data, aes(x = date, y = evaporation_from_open_water_surfaces_excluding_oceans_sum, colour = "Test sample"), size = 1) +
  geom_line(data = plot_data, aes(x = date, y = y_pred_rf, colour = "Predictions"), size = 0.8) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))

sum((plot_data$evaporation_from_open_water_surfaces_excluding_oceans_sum-plot_data$y_pred_rf)^2)

#############
#--XGBOOST--#
#############
# Now we onehot encode the month and day
categorical_vars <- sapply(train_data, is.factor)
train_encoded <- cbind(train_data[, !categorical_vars, drop = FALSE], model.matrix(~ . - 1, train_data[, categorical_vars]))
categorical_vars <- sapply(test_data, is.factor)
test_encoded <- cbind(test_data[, !categorical_vars, drop = FALSE], model.matrix(~ . - 1, test_data[, categorical_vars]))

# Setup train data to be understood by xgboost
X_train <- train_encoded[, -1, drop = FALSE]
y_train <- train_encoded[, 1]
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)

# Setup data to be understood by xgboost
X_test <- test_encoded[, -1, drop = FALSE]
y_test <- test_encoded[, 1]
dtest <- xgb.DMatrix(data = as.matrix(X_test))

# Setup cross validation parameters
ctrl <- trainControl(method = "cv", number = 5)  # You can adjust the number of folds

# Set the parameters to tune for xgBoost
param_grid <- data.frame(
  nrounds = c(100, 500, 1000),                 # Adjust these values based on your preferences
  max_depth = c(6, 9, 15),                     # Max Tree Depth
  eta = c(0.01, 0.1, 0.3),                    # Shrinkage
  gamma = c(0, 1, 2),                         # Minimum Loss Reduction
  subsample = c(0.7, 0.8, 0.9),               # Subsample Percentage
  colsample_bytree = c(0.7, 0.8, 0.9),       # Subsample Ratio of Columns               # Prob. of Skipping Drop-out
  min_child_weight = c(1, 3, 5)               # Minimum Sum of Instance Weight
)

xgboostCrossValidation <- train(
  formula,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = param_grid
)

xgboostCrossValidation$bestTune

params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 9,
  gamma = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)


# Train the best model
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 1000, verbose = 0)

# Feature importance
importance_matrix <- xgb.importance(
  feature_names = colnames(dtrain), 
  model = xgb_model
)
xgb.plot.importance(importance_matrix)

# Make predictions on the test data
y_pred_xgb <- predict(xgb_model, newdata = dtest)

# Assuming you have vectors y_test and y_pred, and the month information
# Create a data frame
plot_data <- cbind(data_for_plot, y_pred_xgb)


# Plot the series
ggplot() +
  geom_line(data = plot_data, aes(x = date, y = evaporation_from_open_water_surfaces_excluding_oceans_sum, colour = "Test sample"), size = 1) +
  geom_line(data = plot_data, aes(x = date, y = y_pred_xgb, colour = "Predictions"), size = 0.8) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))

sum((plot_data$evaporation_from_open_water_surfaces_excluding_oceans_sum-plot_data$y_pred_xgb)^2)

#########
#--GBM--#
#########

# Setup cross validation parameters
ctrl <- trainControl(method = "cv", number = 5)  # You can adjust the number of folds

# Set the parameters to tune for xgBoost
param_grid <- data.frame(
  n.trees = c(100, 300, 500, 1000, 3000), 
  interaction.depth = c(4, 5, 6, 7, 8), 
  shrinkage = c(0.001, 0.01, 0.03, 0.1, 0.05), 
  n.minobsinnode = c(1, 2, 3, 4, 5)             
)

gbmCrossValidation <- train(
  formula,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = param_grid,
  verbose = 0
)

gbmCrossValidation$bestTune


gbm_model <- gbm(formula, 
                 data = train_data, 
                 distribution = "gaussian",
                 n.trees = 500, 
                 interaction.depth = 6,
                 shrinkage = 0.03, 
                 n.minobsinnode = 3)

y_pred_gbm <- predict(gbm_model, newdata = test_data)  # Replace best_iteration with the appropriate value

# Assuming you have vectors y_test and y_pred, and the month information
# Create a data frame
plot_data <- cbind(data_for_plot, y_pred_gbm)

# Plot the series
ggplot() +
  geom_line(data = plot_data, aes(x = date, y = evaporation_from_open_water_surfaces_excluding_oceans_sum, colour = "Test sample"), size = 1) +
  geom_line(data = plot_data, aes(x = date, y = y_pred_gbm, colour = "Predictions"), size = 0.8) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))


sum((plot_data$evaporation_from_open_water_surfaces_excluding_oceans_sum-plot_data$y_pred_gbm)^2)
