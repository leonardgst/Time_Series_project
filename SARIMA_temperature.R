# In this file we fit a SARIMA model to the temperature variable.
library(dplyr)
library(knitr)
library(lubridate)
library(tidyr)
library(ggplot2)
library(cowplot) 
library(xts)
library(TSA)
library(tseries)
library(caschrono)
library(forecast)
library(gbm)

# We import the data that we have downloaded with python and google earth engine
rm(list=ls())
data4 <- read.csv("data_temperature_2013_2023.csv")
data3 <- read.csv("data_temperature_2003_2013.csv")
data2 <- read.csv("data_temperature_1993_2003.csv")
data1 <- read.csv("data_temperature_1983_1993.csv")

# We're doing some preprocessing on the data to select only the important variables
data <- rbind(data1, data2, data3, data4)
data <- data %>% arrange(date)
data$date <- as.Date(data$date, format="%Y-%m-%d")
data <- data %>% select(-X)
data <- data %>% select(-year)
data <- data %>%
  mutate(year = lubridate::year(date),
         month = lubridate::month(date, label = FALSE))
# Here we transform daily data to monthly data
data <- data %>%
  group_by(year, month) %>%
  summarise(temperature = mean(moyenne_temperature_celsius, na.rm = TRUE))
data$date <- paste(data$year,data$month, sep = "-")
data$date <- as.Date(paste0(data$date, "-1"), format="%Y-%m-%d")
data <- data %>% select(-month)
data <- data %>% select(-year)

# We plot the temperature over time
ggplot(data, aes(x = date, y = temperature)) +
  geom_line() +
  labs(
       x = "Date",
       y = "Temperature (°C)")

# We plot the ACF of both regular series and differenced series
ts_data <- ts(data$temperature, frequency = 12)
ts_data_12 <-diff(ts_data, 12)
plot2acf(ts_data,ts_data_12,lag.max=45,main=c("temperature",expression(paste("(1-",B^{12},") temperature",sep=""))))


#We plot here the decomposition of the time series
ts_data_decomposed <- decompose(ts_data)
plot(ts_data_decomposed)

# The following is going to be used to choose the parameters of the SARIMA model
# Plot ACF and PACF for original time series (ts_data)
par(mfrow=c(2,2))
acf(ts_data, main="ACF - Original Time Series (ts_data)")
pacf(ts_data, main="PACF - Original Time Series (ts_data)")

# Plot ACF and PACF for differenced series (ts_data_12)
acf(ts_data_12, main="ACF - Differenced Series (ts_data_12)")
pacf(ts_data_12, main="PACF - Differenced Series (ts_data_12)")

#We partition the dataset into training and test samples 
train_set <- subset(data, year >= 1984 & year <= 2015)
test_set <- subset(data, year >= 2016 & year <= 2022)

# We scale and reduce both samples. We scale with the mean and variance of the training dataset
# to avoid tacking information from the testing sample. 
train_set_mean <- mean(train_set$temperature)
train_set_sd <- sd(train_set$temperature)
train_set$temperature<- (train_set$temperature-train_set_mean)/train_set_sd
test_set$temperature<- (test_set$temperature-train_set_mean)/train_set_sd


#The first Arima model with visual observations of the ACF and PACF 
model <- Arima(train_set$temperature, order= c(1,0,0), seasonal = list(order=c(1,1,0), periods=12),method = "CSS", include.constant=TRUE)
summary(model)
Box.test(model$residuals, lag=24, type = "Ljung-Box")
#The residuals are not good. 

# We're going to use this function that we have founded in "Introductory Time Series
# with R" by Paul S.P. Cowpertwait · Andrew V. Metcalfe
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
                   seasonal = list(order=c(P,D,Q), periods=12),
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
  list(best.model)
}

# We use this function to get the best parameters 
param <- get.best.arima(train_set$temperature, maxord = c(3,3,3,3,3,3))
model2 <- Arima(train_set$temperature, order= c(2,0,1), seasonal = list(order=c(3,1,2), periods=12),method = "CSS", include.constant=TRUE)
summary(model2)
Box.test(model2$residuals, lag=24, type = "Ljung-Box")

# We retry but with a larger q
param <- get.best.arima(train_set$temperature, maxord = c(3,3,15,3,3,3))
model3 <- Arima(train_set$temperature, order= c(2,0,14), seasonal = list(order=c(3,1,3), periods=12),method = "CSS", include.constant=TRUE)
autoplot(model3)
Box.test(model3$residuals, lag=24, type = "Ljung-Box")
summary(model3)

# We verify here that the residuals are good with QQ-plot and Shapiro Test 
residuals <- residuals(model3)
par(mfrow=c(1,1))
qqnorm(residuals, main="Q-Q Plot des Résidus")
qqline(residuals)
shapiro.test(residuals)
acf(residuals)
pacf(residuals)

#Here, we're doing predictions to compare to other models
test_predictions <- forecast(model3, h = nrow(test_set))
pred <- test_predictions$mean
MSE <- sum((test_set$temperature - pred)^2)

# here we do a plot to comapre the predictions with the the test samples
ggplot() +   
  geom_line(aes(x = test_set$date, y = test_set$temperature, colour = "Test sample"), size = 0.8) + 
  geom_line(aes(x = test_set$date, y = pred, colour = "Predictions"), size = 1) +   
  theme_minimal() +   
  labs(y = "Temperature", x = "Date") +   
  scale_color_manual(name = "Time series", values = c("Test sample" = "black", "Predictions" = "red")) 

# We save the file with the predictions to add to the water evaporation model
write.csv(x = pred, file="y_pred_temp_sarima.csv")
