# In this file you can perform machine lerning models for the evaporation time series analysis.
rm(list=ls())
library(randomForest)
library(caret)
source("importation_datasets.R")

# Get ride of unwanted features
evaporation <- evaporation[,-1]
evaporation <- evaporation[,-3]
# Set date as date object
evaporation$date <- as.Date(evaporation$date)
# Groupby month
evaporation <- evaporation %>% # Group by month
  group_by(date = lubridate::floor_date(date, 'month')) %>%
  summarize(evap_month = sum(evaporation_from_open_water_surfaces_excluding_oceans_sum))

# Adding year and month
evaporation$year <- year(evaporation$date)
evaporation$month <- month(evaporation$date)

# We want to create lag for the previous day, the mean of the last 7 and 30 days and the same day one year ago
evaporation$lag1 <- lag(evaporation$evap_month, 1)
evaporation$lag2 <- lag(evaporation$evap_month, 2)
evaporation$lag3 <- lag(evaporation$evap_month, 3)
evaporation$lag12 <- lag(evaporation$evap_month, 12)

# We want to retain only the non NA part ie we lose a year. But it is done by keeping only from 1984.
# Splitting the data into test and train samples
train_data <- subset(evaporation, year >= 1984 & year <= 2015)
test_data <- subset(evaporation, year >= 2016 & year <= 2022)
# Getting ride of unwanted features
train_data <- train_data[,-1]
train_data <- train_data[,-2]
# Kepping this dataframe for the plot at the end
data_for_plot <- test_data
# Getting ride of unwanted features
test_data <- test_data[,-1]
test_data <- test_data[,-2]

# Plotting our data
ggplot(data_for_plot, aes(x=date, y=evap_month)) +
  geom_line() +
  theme_minimal() +
  labs(y = "Total evaporation (meter water equivalent)", x = "Date (mois)")

# Setting the month as a factor for better incorporation into the ML models
train_data$month <- as.factor(train_data$month)
test_data$month <- as.factor(test_data$month)

# Normalizing y and the features
y_train_mean <- mean(train_data$evap_month)
y_train_sd <- sd(train_data$evap_month)
train_data$evap_month <- (train_data$evap_month-y_train_mean)/y_train_sd
test_data$evap_month <- (test_data$evap_month-y_train_mean)/y_train_sd
data_for_plot$evap_month <- (data_for_plot$evap_month-y_train_mean)/y_train_sd

lag1_train_mean <- mean(train_data$lag1)
lag1_train_sd <- sd(train_data$lag1)
train_data$lag1 <- (train_data$lag1-lag1_train_mean)/lag1_train_sd
test_data$lag1 <- (test_data$lag1-lag1_train_mean)/lag1_train_sd

lag2_train_mean <- mean(train_data$lag2)
lag2_train_sd <- sd(train_data$lag2)
train_data$lag2 <- (train_data$lag2-lag2_train_mean)/lag2_train_sd
test_data$lag2 <- (test_data$lag2-lag2_train_mean)/lag2_train_sd

lag3_train_mean <- mean(train_data$lag3)
lag3_train_sd <- sd(train_data$lag3)
train_data$lag3 <- (train_data$lag3-lag3_train_mean)/lag3_train_sd
test_data$lag3 <- (test_data$lag3-lag3_train_mean)/lag3_train_sd

lag12_train_mean <- mean(train_data$lag12)
lag12_train_sd <- sd(train_data$lag12)
train_data$lag12 <- (train_data$lag12-lag12_train_mean)/lag12_train_sd
test_data$lag12 <- (test_data$lag12-lag12_train_mean)/lag12_train_sd

###################
#--RANDOM FOREST--#
###################
## Pre process for Random Forest
# Specify the target variable
target_variable <- "evap_month"
# Specify the predictor variables (features)
predictor_variables <- setdiff(names(train_data), target_variable)
# Set up the formula
formula <- as.formula(paste(target_variable, "~", paste(predictor_variables, collapse = "+")))

## Setting things up for the cross validation
param_grid <- data.frame(
  mtry = c(2, 4, 6, 8, 10)
)
ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
# Performing cross validation
randomforestCrossValidation <- train(
  formula,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneGrid = param_grid,
  ntree = 5000
)

# Retrieving the best parameters
randomforestCrossValidation$bestTune

# Performing Random Forest with the best parameters
randomforestModel <- randomForest(
  formula,
  data = train_data,
  ntree = 1000,            
  mtry = 4 # You should adjust the mtry
)

# Make predictions on the test set
y_pred <- predict(randomforestModel, newdata = test_data)

# Get feature importance
importance_matrix <- importance(randomforestModel)
# Plot feature importance
varImpPlot(randomforestModel)

# Create a data frame
plot_data <- cbind(data_for_plot, y_pred)

# Plot the series
ggplot() +
  geom_line(data = plot_data, aes(x = date, y = evap_month, colour = "Test sample"), size = 1) +
  geom_line(data = plot_data, aes(x = date, y = y_pred, colour = "Predictions"), size = 0.8) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))

# Compute the empirical risk
sum((plot_data$evap_month-plot_data$y_pred)^2)

#############
#--XGBoost--#
#############
# Now we onehot encode the month and day
categorical_vars <- sapply(train_data, is.factor)
train_encoded <- cbind(train_data[, !categorical_vars, drop = FALSE], model.matrix(~ . - 1, train_data[, categorical_vars]))
test_encoded <- cbind(test_data[, !categorical_vars, drop = FALSE], model.matrix(~ . - 1, test_data[, categorical_vars]))

# Setup train data to be understood by xgboost
X_train <- train_encoded[, -1, drop = FALSE]
y_train <- train_encoded[, 1]
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)

# Setup data to be understood by xgboost
X_test <- test_encoded[, -1, drop = FALSE]
y_test <- test_encoded[, 1]
dtest <- xgb.DMatrix(data = as.matrix(X_test))

# Specify the target variable
target_variable <- "evap_month"

# Specify the predictor variables (features)
predictor_variables <- setdiff(names(train_data), target_variable)

# Set up the formula
formula <- as.formula(paste(target_variable, "~", paste(predictor_variables, collapse = "+")))

# Setup cross validation parameters
ctrl <- trainControl(method = "cv", number = 5)  # You can adjust the number of folds

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

xgboostCrossValidation <- train(
  formula,
  data = train_data,
  method = "xgbDART",
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
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 500, verbose = 0)

# Feature importance
importance_matrix <- xgb.importance(
  feature_names = colnames(dtrain), 
  model = xgb_model
)
xgb.plot.importance(importance_matrix)

# Make predictions on the test data
y_pred <- predict(xgb_model, newdata = dtest)

# Assuming you have vectors y_test and y_pred, and the month information
# Create a data frame
plot_data <- cbind(data_for_plot, y_pred)


# Plot the series
ggplot() +
  geom_line(data = plot_data, aes(x = date, y = evap_month, colour = "Test sample"), size = 1) +
  geom_line(data = plot_data, aes(x = date, y = y_pred, colour = "Predictions"), size = 0.8) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))

sum((plot_data$evap_month-plot_data$y_pred)^2)


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
  geom_line(data = plot_data, aes(x = date, y = evap_month, colour = "Test sample"), size = 1) +
  geom_line(data = plot_data, aes(x = date, y = y_pred_gbm, colour = "Predictions"), size = 0.8) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))


sum((plot_data$evap_month-plot_data$y_pred_gbm)^2)



