
# =====================================================================================
# PART 0: DATA DOWNLOADING
#
# This script is for downloading and processing all data for our prediction model.
# Here we download, merge and transform the datasets both from Kaggle competition
# and for constructing our own variables using information from different sources.
# 
# 1) Good practices & libraries & Import kaggle datasets
# 2) Analize Kaggle datasets
# 3) 
# 4) 
# 5) 
# 6) 
# =====================================================================================

# =====================================================================================
# 1) Good practices & libraries & Import kaggle datasets
# =====================================================================================

# Clean variables
rm(list = ls())

# DO NOT FORGET TO SET YOU OWN DIRECTORRY
# setwd("C:/Users/Asuar/OneDrive/Escritorio/Libros Clases/Economía/Big Data/Problem-Set-3-Making-Money-with-ML")

# Libraries
require("pacman")
p_load("tidyverse", "dplyr") #TODO

# Import Kaggle datasets
train <- read.csv(unz("data/uniandes-bdml-2025-20-ps-3.zip", "train.csv"))
test <- read.csv(unz("data/uniandes-bdml-2025-20-ps-3.zip", "test.csv"))


# =====================================================================================
# 1) Analize Kaggle datasets
# =====================================================================================

# Values suspected of only 1 type (city, year, property_type, operation_type)
unique(train$city) # all Bogota
unique(test$city)
unique(train$year) # 2019, 2020, 2021
unique(test$year)
unique(train$property_type) # Casa, Apartamento
unique(test$property_type)
unique(train$operation_type) # all Venta
unique(test$operation_type)

# Remove simple variables
train <- train %>% select(-city, -operation_type)
test <- test %>% select(-city, -operation_type)

# Check missing percentage
na_report <- tibble(
  variable = names(train),
  train = map_dbl(train, ~ round(mean(is.na(.)) * 100, 2)),
  test  = map_dbl(test,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train))

na_report











