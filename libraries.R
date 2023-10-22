##########################################################
##########################################################
##                                                      ##
##                   EXPLORATION FILE                   ##
##                                                      ##
##########################################################
##########################################################

#### LIBRARIES ####

library(dplyr)
library(ggplot2)


#### DATASET IMPORT ####
# ad_viz_plotval_data_2021.csv = Carbonne dans New-York depuis 2021
#aqs_sites.csv = jspatrop

test <- rbind(read.csv("ad_viz_plotval_data_2021.csv"),
              read.csv("ad_viz_plotval_data_2022.csv"),
              read.csv("ad_viz_plotval_data_2023.csv"))


