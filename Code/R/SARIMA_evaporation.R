# In this file you can perform the time series analysis for the evaporation data.

rm(list = ls())
library(tidyverse)
library(ggplot2)
library(forecast)
library(tseries)
library(lmtest)
source("importation_dataset.R")


# Get ride of the two unwanted columns
evaporation <- evaporation[,-1]
evaporation <- evaporation[,-3]

# Transforms the date into a date object
evaporation$date <- as.Date(evaporation$date)

# We sum the total evaporation per month
evap_montly <- evaporation %>% # Group by month
  group_by(month = lubridate::floor_date(date, 'month')) %>%
  summarize(evap_month = sum(evaporation_from_open_water_surfaces_excluding_oceans_sum))

# Splitting the data into test and train samples
evap_montly$year <- year(evap_montly$month)
evap_montly$month <- month(evap_montly$month)
# The train sample is from 1984 to 2015 and test sample from 2016 to 2022
train_data <- subset(evap_montly, year >= 1980 & year <= 2010)
test_data <- subset(evap_montly, year >= 2011 & year <= 2022)
# Again get ride of unwanted variable
train_data <- train_data[,-3]
test_data <- test_data[,-3]

# Normalizing the data
train_mean <- mean(train_data$evap_month)
train_sd <- sd(train_data$evap_month)
train_data$evap_month <- (train_data$evap_month-train_mean)/train_sd

# This function looks for the best parameters (p, d, q)(P, D, Q)
# with parameters inferior or equal to the ones given
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
                   include.constant = TRUE)
      
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

# We create a ts object from our data
y <- ts(data = train_data$evap_month, start = c(1980, 1), end = c(2011, 12), frequency = 12)

# We first plot the serie
ggplot(train_data, aes(x=month, y=evap_month)) +
  geom_line() +
  theme_minimal() +
  labs(y = "Total evaporation (meter water equivalent)", x = "Date (mois)")

# Looking at the trend
plot(decompose(y))
adf.test(y)

# Plot by year to see if there is a trend
seasonplot(y, xlab = "Months", ylab = "Total evaporation every year", main = "")

# Plotting ACF and PACF
acf_y <- acf(y, plot = FALSE)
plot(acf_y, main = "")
pacf_y <- pacf(y, plot = FALSE)
plot(pacf_y, main = "")

# Box-Cox : If near one no transformation needed, else you should transform your data to (old_data)^lambda
BoxCox.lambda(y)

# We are going to test and select the best arima model for parameters >=2
bestArima <- get.best.arima(y, maxord = c(2,2,2,2,2,2))

# The model has parameters
bestArima[[3]]

# We retrain the model
myModel <- Arima(y, order = c(0, 0, 0), seasonal = c(0, 1, 1), lambda = NULL, include.constant = TRUE)

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

# Plot of the predictions of the model
d.forecast <- forecast(myModel, level = c(95), h = 1000)
autoplot(d.forecast)

# Generate forecasts for the test data
test_predictions <- forecast(myModel, h = nrow(test_data))
test_data$pred <- test_predictions$mean

# Must also scale the test sample
test_data$evap_month <- (test_data$evap_month-train_mean)/train_sd

# Plot the series
ggplot() +
  geom_line(data = test_data, aes(x = month, y = evap_month, colour = "Test sample"), size = 0.8) +
  geom_line(data = test_data, aes(x = month, y = pred, colour = "Predictions"), size = 1) +
  theme_minimal() +
  labs(y = "Total evaporation", x = "Date") +
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red"))

sum((test_data$evap_month-test_data$pred)^2)




