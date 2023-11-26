##########################################################
##########################################################
##                                                      ##
##                   EXPLORATION FILE                   ##
##                                                      ##
##########################################################
##########################################################

#### LIBRARIES ####

library(readxl)
library(dplyr)
library(ggplot2)
library(forecast)
library(TSA)
library(caschrono)
library(tseries)
library(fracdiff)


#### DATASET IMPORT ####
rm(list=ls())

raw_db <- read_xlsx("N_ice.xlsx")

db <- raw_db %>% 
  mutate(Date = as.Date(paste(Year, Month, Day, sep = "/"))) %>% 
  select(Extent, Date, Year, Month)
db$Extent <- as.numeric(db$Extent)
summary(db)



extent_ts <- ts(db$Extent, frequency = 1, start = c(db$Year, db$Month))

ggplot(data = db, aes(x = Date, y = Extent)) +
  geom_line() +
  labs(x = "Date", y = "Extent") +
  ggtitle("Série temporelle de l'Extent")

auto.arima(x = extent_ts)

fit <- arima(extent_ts, order = c(5, 1, 1))
AIC(fit) 
# Affichage des graphiques ACF et PACF pour le modèle ajusté
acf(residuals(fit))
pacf(residuals(fit))
acf(extent_ts)
pacf(extent_ts)
# peut etre utiliser un FARIMA car dependance tres loin dans le passé

t(Box.test.2(extent_ts, nlag=2:5, type = "Ljung-Box", decim = 2))
adf.test(extent_ts)


library(fracdiff)

# Estimation du paramètre d pour le modèle FARIMA
d_est <- fracdiff(diff(log(extent_ts)), h = 1)$d

# Ajustement du modèle FARIMA
farima_fit <- Arima(extent_ts, order = c(5, 1, 1), seasonal = list(order = c(0, 0, 0), period = 1), fractional = d_est)

# Affichage du résumé du modèle FARIMA
summary(farima_fit)

# Affichage des graphiques ACF et PACF pour les résidus du modèle FARIMA
acf(residuals(farima_fit))
pacf(residuals(farima_fit))


