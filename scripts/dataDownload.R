
# =====================================================================================
# PART 0: DATA DOWNLOADING
#
# This script is for downloading and processing all data for our prediction model.
# Here we download, merge and transform the datasets both from Kaggle competition
# and for constructing our own variables using information from different sources.
# 
# 1) Good practices & libraries & Import kaggle datasets
# 2) Analize Kaggle datasets
# 3) New variables and data imputation using "Title" and "Description"
#   3.1) Apto
#   3.2) Casa
#   3.3) Merge new variables with original dataset
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
p_load("tidyverse", "dplyr", "tidyr", "tm", "stringr") #TODO

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

# Distribución entre casas y aptos
  # En train 75,5% apto, en test 97,3%
list(train = train, test = test) %>%
  bind_rows(.id = "dataset") %>%
  group_by(dataset) %>%
  mutate(total = n()) %>%          # total de observaciones por dataset
  group_by(dataset, property_type, total) %>%
  summarise(
    pct = round(100 * n() / first(total), 2),
    .groups = "drop"
  ) %>%
  select(-total) %>%
  pivot_wider(
    names_from = dataset,
    values_from = pct,
    names_prefix = "pct_"
  ) %>%
  arrange(property_type)

# Description size distribution (TODO: SOLO PARA VER, LUEGO SE QUITA)
train %>%
  mutate(largo = nchar(description)) %>%
  ggplot(aes(x = largo)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Desctiption size distribution (# characters) in train",
    x = "# of characters", y = "Frequency"
  )

test %>%
  mutate(largo = nchar(description)) %>%
  ggplot(aes(x = largo)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Desctiption size distribution (# characters) in test",
    x = "# of characters", y = "Frequency"
  )


# =====================================================================================
# TODO:ideas de qué hacer con los datos
# =====================================================================================

# Ideas: 
# - # de baños missing
# - # metros cuadrados
# - si tiene parqueadero y cuantos
# - ver si fue remodelado
# - Balcon? solo dummy o por número?
# - algunos dicen si tienen area de ninos, puede crearse la dummy
# - diferenciar por año (pre, durante y post pandemia cambia mucho)
# - Ver si puedo sacar edad por grupos (+20 años no dan credito vivienda)
# - si tiene depósito (en aptos)
# - # de piso (en aptos)
# - por (lat, long) y titulo se puede sacar sector (en ese orden)
    # intentar graficar en mapa para saber qué sectores hay
    # por zona se puede sacar estadísticas de crímenes (robos, asaltos, etc)
    # Se pueden buscar datos de tráfico, también de valor promedio por zona
    # Se puede buscar datos de disponibilidad de transporte público
    # Buscar valor arriendo promedio, estrato promedio, valor servicios
    # si hay cc (centro comercial) cerca; incluso distancia al más cercano
# - varias descripciones tienen mencionan si hay parques al lado
# - hay duplicados en (lat,long), implicando mismo edificio
    # puede usarse para imputar datos
    # Hay apartamentos con diferente id, pero es un dato repetido


# HAY DATOS ENTRE TEST Y TRAIN REPETIDOS POR (LAT,LONG), TITULO Y DESCRIPCION
# Emparejar por lat y long para encontrar coincidencias cruzadas
cross_dups <- inner_join(
  train %>% select(property_id, lat, lon),
  test  %>% select(property_id, lat, lon),
  by = c("lat", "lon"),
  suffix = c("_train", "_test")
)
# Mostrar resultado
cross_dups


train %>%
  filter(property_id == "e0168044ca32b35cb057ab10") %>%
  select(property_id, lat, lon, year, title, description)

test %>%
  filter(property_id == "53d33316e237830ad463a26b") %>%
  select(property_id, lat, lon, year, title, description)


# =====================================================================================
# New variables and data imputation using "Title" and "Description" (Apto)
# =====================================================================================

# Spit data by property type, create new variables and imputate missings in other, 
# at the end reunify all into original dataset

# TODO: Terminar cuando estén todas las nuevas variables
# Create new variables (parqueaderos, piso)
train <- train %>% mutate(parqueaderos = NA_real_, num_piso = NA_real_, remodelado = NA_real_,
                          deposito = NA_real_, balcon = NA_real_)
test <- test %>% mutate(parqueaderos = NA_real_, num_piso = NA_real_, remodelado = NA_real_,
                          deposito = NA_real_, balcon = NA_real_)

# =====================================================================================
# 3.1) Apto
# =====================================================================================

# Only apartments
train_apto <- train %>% filter(property_type == "Apartamento")
test_apto <- test %>% filter(property_type == "Apartamento")

# Function to clean data
clean_corpus <- function(text_vector) {
  
  corpus <- Corpus(VectorSource(text_vector))
  
  # 🔹 1. General transformations
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, content_transformer(removePunctuation))
  corpus <- tm_map(corpus, content_transformer(stripWhitespace))
  
  # 🔹 2. Sinonims replacement (with regular expresions)
  replace_synonyms <- content_transformer(function(x) {
    x <- str_replace_all(x, "\\b(parqueaderos|garaje|garajes|estacionamiento)\\b", "parqueadero")
    x <- str_replace_all(x, "\\b(bañ|banos|baos|bao)\\b", "bano")
    x <- str_replace_all(x, "\\b(habitaciones|alcobas|alcoba)\\b", "habitacion")
    x <- str_replace_all(x, "\\b(depsito)\\b", "deposito")
    x <- str_replace_all(x, "\\b(balcn|terraza|balcones|balcons)\\b", "balcon")
    x <- str_replace_all(x, "\\b(remodele)\\b", "remodelado")
    return(x)
  })
  corpus <- tm_map(corpus, replace_synonyms)
  
  # 🔹 3. Change numbers from words to digits
  replace_numbers <- content_transformer(function(x) {
    x <- str_replace_all(x, "\\b(uno|una|un|primer|primero|1er|1ero)\\b", "1")
    x <- str_replace_all(x, "\\b(dos|segunda|segundo|2ndo)\\b", "2")
    x <- str_replace_all(x, "\\b(tres|tercer|tercero|3ro)\\b", "3")
    x <- str_replace_all(x, "\\b(cuatro|cuarto|4to)\\b", "4")
    x <- str_replace_all(x, "\\b(cinco|quinto|5to)\\b", "5")
    x <- str_replace_all(x, "\\b(seis|sexto|6to)\\b", "6")
    x <- str_replace_all(x, "\\b(siete|septimo|7mo)\\b", "7")
    x <- str_replace_all(x, "\\b(ocho|octavo|8vo)\\b", "8")
    x <- str_replace_all(x, "\\b(nueve|noveno|9no)\\b", "9")
    x <- str_replace_all(x, "\\b(diez|decimo|10mo)\\b", "10")
    x <- str_replace_all(x, "\\b(once|onceavo|11vo)\\b", "11")
    x <- str_replace_all(x, "\\b(doce|doceavo|12vo)\\b", "12")
    
    return(x)
  })
  corpus <- tm_map(corpus, replace_numbers)
  
  return(corpus)
}

# Corpus (still with stopwords) for "Apartamento" TODO: usar title tal vez no sea necesario
corpus_train_apto_desc <- clean_corpus(train_apto$description)
corpus_test_apto_desc <- clean_corpus(test_apto$description)
corpus_train_apto_title <- clean_corpus(train_apto$title)
corpus_test_apto_title <- clean_corpus(test_apto$title)

# ------------------------------------------------------------------------------

# See missing % BEFORE imputation
na_report <- tibble(
  variable = names(train_apto),
  train_apto = map_dbl(train_apto, ~ round(mean(is.na(.)) * 100, 2)),
  test_apto  = map_dbl(test_apto,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train_apto))

na_report

# Data imputation function
impute_from_description <- function(df, corpus_desc) {
  
  # 1. Convert corpus object into normal text
  desc_text <- sapply(corpus_desc, as.character)
  
  # 2. Generate local copies of variables before imputation
  bathrooms_new      <- df$bathrooms
  rooms_new          <- df$rooms
  surface_total_new  <- df$surface_total # TODO no sé si es esta o "surface_covered"
  
  # -------------------------------------------------------------------
  # 3. IMPUTE BATHROOMS — STEP 1: extract explicit numeric patterns
  # Look for expressions like "2 bano"
  found_bath <- str_extract(desc_text, "\\b([0-9]+)\\s*(bano)\\b")
  found_bath_num <- as.numeric(str_extract(found_bath, "[0-9]+"))
  
  # Impute missing values using explicit numbers
  bathrooms_new <- ifelse(is.na(bathrooms_new) & !is.na(found_bath_num),
                      found_bath_num, bathrooms_new)
  
  # -------------------------------------------------------------------
  # 4. IMPUTE BATHROOMS — STEP 2: count occurrences if still missing
  # Count total mentions of "baño"/"bano" variants when no explicit number is found
  bath_count <- str_count(desc_text, "\\b(bano)\\b")
  
  # Use the count only for those that are still missing
  bathrooms_new <- ifelse(is.na(bathrooms_new) & bath_count > 0, bath_count, bathrooms)
  
  # -------------------------------------------------------------------
  # 5. IMPUTE ROOMS
  # Look for sentences like "# habitacion"
  found_rooms <- str_extract(desc_text, "\\b([0-9]+)\\s*(habitacion)\\b")
  found_rooms_num <- as.numeric(str_extract(found_rooms, "[0-9]+"))
  
  rooms_new <- ifelse(is.na(rooms_new) & !is.na(found_rooms_num),
                  found_rooms_num, rooms_new)
  
  # -------------------------------------------------------------------
  # 6. IMPUTE SIZE (m²)
  # Look for sentences like "# mts", "# mts2", and so on
  found_size <- str_extract(desc_text, "\\b([0-9]+)\\s*(mts|mts2|m2|metros)\\b")
  found_size_num <- as.numeric(str_extract(found_size, "[0-9]+"))
  
  surface_total_new <- ifelse(is.na(surface_total_new) & !is.na(found_size_num),
                 found_size_num, surface_total_new)
  
  # -------------------------------------------------------------------
  # 7. Imputated values in the apto dataset
  df <- df %>%
    mutate(
      bathrooms     = bathrooms_new,
      rooms         = rooms_new,
      surface_total = surface_total_new
    )
  
  return(df)
}

# Impute data
train_apto <- impute_from_description(train_apto, corpus_train_apto_desc)
test_apto <- impute_from_description(test_apto, corpus_test_apto_desc)

# See missing % AFTER imputation
na_report <- tibble(
  variable = names(train_apto),
  train_apto = map_dbl(train_apto, ~ round(mean(is.na(.)) * 100, 2)),
  test_apto  = map_dbl(test_apto,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train_apto))

na_report

# ------------------------------------------------------------------------------

# Function to create new features 
create_extra_features <- function(df, corpus_desc) {
  
  # 1. Convert corpus object into normal text
  desc_text <- sapply(corpus_desc, as.character)
  
  # 2. Generate local copies of new variables
  parqueadero_new <- df$parqueaderos
  num_piso_new    <- df$num_piso
  deposito_new    <- df$deposito
  remodelado_new  <- df$remodelado
  balcon_new      <- df$balcon
  
  # -------------------------------------------------------------------
  # 3. IMPUTE PARQUEADERO
  # Look for patterns like "2 parqueaderos", "parqueadero 1", or "1 parqueadero"
  found_park <- str_extract(desc_text, "([0-9]+)\\s*parqueadero")
  found_park_num <- as.numeric(str_extract(found_park, "[0-9]+"))
  
  # If no number but the word exists, assume 1
  has_park_word <- str_detect(desc_text, "parqueadero")
  
  parqueadero_new <- ifelse(is.na(parqueadero_new) & !is.na(found_park_num),
                            found_park_num,
                            parqueadero_new)
  parqueadero_new <- ifelse(is.na(parqueadero_new) & has_park_word, 1, parqueadero_new)
  
  # -------------------------------------------------------------------
  # 4. IMPUTE NUM_PISO
  # Look for "piso 3" or "3 piso"
  found_piso <- str_extract(desc_text, "(piso\\s*[0-9]+)|([0-9]+)\\s*piso")
  found_piso_num <- as.numeric(str_extract(found_piso, "[0-9]+"))
  
  num_piso_new <- ifelse(is.na(num_piso_new) & !is.na(found_piso_num),
                         found_piso_num,
                         num_piso_new)
  
  # -------------------------------------------------------------------
  # 5. IMPUTE DEPOSITO
  # If the text mentions "deposito" at least once, assume 1
  has_deposito <- str_detect(desc_text, "deposito")
  deposito_new <- ifelse(is.na(deposito_new) & has_deposito, 1, deposito_new)
  
  # -------------------------------------------------------------------
  # 6. IMPUTE REMODELADO
  # If the text mentions "remodelado", assume 1
  has_remodelado <- str_detect(desc_text, "remodelado")
  remodelado_new <- ifelse(is.na(remodelado_new) & has_remodelado, 1, remodelado_new)
  
  
  # -------------------------------------------------------------------
  # 7. IMPUTE BALCON
  found_balcon <- str_extract(desc_text, "([0-9]+)\\s*balcon")
  found_balcon_num <- as.numeric(str_extract(found_balcon, "[0-9]+"))
  
  # If no number but the word exists, assume 1
  has_balcon_word <- str_detect(desc_text, "balcon")
  
  balcon_new <- ifelse(is.na(balcon_new) & !is.na(found_balcon_num),
                            found_balcon_num,
                            balcon_new)
  balcon_new <- ifelse(is.na(balcon_new) & has_balcon_word, 1, balcon_new)
  
  # -------------------------------------------------------------------
  # 8. Update dataframe with imputed values
  df <- df %>%
    mutate(
      parqueaderos = parqueadero_new,
      num_piso     = num_piso_new,
      deposito     = deposito_new,
      remodelado   = remodelado_new,
      balcon       = balcon_new
    )
  
  return(df)
}

# Impute data
train_apto <- create_extra_features(train_apto, corpus_train_apto_desc)
test_apto <- create_extra_features(test_apto, corpus_test_apto_desc)

# See missing % AFTER imputation
na_report <- tibble(
  variable = names(train_apto),
  train_apto = map_dbl(train_apto, ~ round(mean(is.na(.)) * 100, 2)),
  test_apto  = map_dbl(test_apto,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train_apto))

na_report


# =====================================================================================
# 3.2) Casa
# =====================================================================================

# Only houses
test_casa <- train %>% filter(property_type == "Casa")
train_casa <- train %>% filter(property_type == "Casa")

# Corpus (still with stopwords) for "Casa" TODO: usar title tal vez no sea necesario
# Para casa posiblemente no sacarlo, pues en testo solo hay 3% de casas
corpus_train_casa_desc <- clean_corpus(train_casa$description)
corpus_test_casa_desc <- clean_corpus(test_casa$description)
corpus_train_casa_title <- clean_corpus(train_casa$title)
corpus_test_casa_title <- clean_corpus(test_casa$title)



# =====================================================================================
# 3.3) Merge new variables with original dataset
# =====================================================================================




