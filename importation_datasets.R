##########################
#  IMPORTATION DATASETS  #
##########################

# ML_evaporation
evaporation <- read.csv("Dataset/daily_Sum_Evap.csv")

# ML_temperature
data_temp <- read.csv("Dataset/Daily_Sum.csv")

# Models_part4.1 and 4.2 and Models_part 4.3
full_data <- read.csv("Dataset/daily_Sum_Evap.csv")
liste_var = c("Dataset/daily_Sum_Temp2M.csv")
y_pred_temp_sarima <- read.csv("Dataset/y_pred_temp_sarima.csv")

# 