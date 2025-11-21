
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
p_load("tidyverse", "dplyr", "tidyr", "tm", "stringr", "text2vec",
       "spacyr", "stopwords", "tokenizers") #TODO

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

# Create variable for "terraza" area
train <- train %>% mutate(terraza = ifelse(!is.na(surface_covered) & !is.na(surface_total) &
                                             surface_covered != surface_total & surface_total > surface_covered, 
                                            surface_total - surface_covered, NA_real_)
                          )
test <- test %>% mutate(terraza = ifelse(!is.na(surface_covered) & !is.na(surface_total) & 
                                           surface_covered != surface_total & surface_total > surface_covered, 
                                            surface_total - surface_covered, NA_real_))

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

# Statistic table of the original data
summary(train)
summary(test)

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
# - falta buscar en descripción si dice "apartamento" o casa y contrastar con property_type
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
# 3) New variables and data imputation using "Title" and "Description" (Apto)
# =====================================================================================

# Spit data by property type, create new variables and imputate missings in other, 
# at the end reunify all into original dataset


# =====================================================================================
# 3.1) Apto
# =====================================================================================

# Only apartments
train_apto <- train %>% filter(property_type == "Apartamento")
test_apto <- test %>% filter(property_type == "Apartamento")

# Function to clean data
clean_corpus <- function(text_vector) {
  
  corpus <- Corpus(VectorSource(text_vector))
  
  # 1. General transformations
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, content_transformer(removePunctuation))
  corpus <- tm_map(corpus, content_transformer(stripWhitespace))
  
  # 2. Sinonims  and writing errors replacement (with regular expresions)
  replace_synonyms <- content_transformer(function(x) {
    x <- str_replace_all(x, "(\\d)([a-zA-Z])", "\\1 \\2") # This because of missing spaces; "2parqueadero"
    x <- str_replace_all(x, "(m|mt|mts|metro|metros)2([0-9]+)", "\\12 \\2") # This for cases like "138 mts23 habitacion 2 bano"
    x <- str_replace_all(x, "(\\d)(m|mt|mt2|mts|m2|mts2|mtrs)(\\d*)", "\\1 \\2\\3") # This for cases like "25m2" or "23mts2"
    
    x <- str_replace_all(x, "\\b(m|mt|mt2|mts|mts2|m2|mtrs2|mtrs)\\b", "metros")
    x <- str_replace_all(x, "\\b(parqueaderos|garaje|garajes|estacionamiento|carro|carros|vehiculo|vehiculos)\\b", "parqueadero")
    x <- str_replace_all(x, "\\b(bañ|baño|banos|baos|bao)\\b", "bano")
    x <- str_replace_all(x, "\\b(habitaciones|alcobas|alcoba|habitacin|habitacon|habitcion)\\b", "habitacion")
    x <- str_replace_all(x, "\\b(depsito)\\b", "deposito")
    x <- str_replace_all(x, "\\b(balcn|balcones|balcons)\\b", "balcon")
    x <- str_replace_all(x, "\\b(remodele|remodelada|remodelado)\\b", "remodelado")
    x <- str_replace_all(x, "\\b(terrazas)\\b", "terraza")
    # The following are common spelling mistakes found
    x <- str_replace_all(x, "\\b(saln)\\b", "salon")
    x <- str_replace_all(x, "\\b(nios)\\b", "ninos")
    x <- str_replace_all(x, "\\b(aos|anio|ano|anos|anios)\\b", "año")
    x <- str_replace_all(x, "\\b(apto)\\b", "apartamento")
    
    return(x)
  })
  corpus <- tm_map(corpus, replace_synonyms)
  
  # 3. Change numbers from words to digits (first 20)
  replace_numbers <- content_transformer(function(x) {
    x <- str_replace_all(x, "\\b(uno|una|un|primer|primero|1er|1ero)\\b", " 1 ")
    x <- str_replace_all(x, "\\b(dos|segunda|segundo|2ndo)\\b", " 2 ")
    x <- str_replace_all(x, "\\b(tres|tercer|tercero|3ro)\\b", " 3 ")
    x <- str_replace_all(x, "\\b(cuatro|cuarto|4to)\\b", " 4 ")
    x <- str_replace_all(x, "\\b(cinco|quinto|5to)\\b", " 5 ")
    x <- str_replace_all(x, "\\b(seis|sexto|6to)\\b", " 6 ")
    x <- str_replace_all(x, "\\b(siete|septimo|spttimo|7mo)\\b", " 7 ")
    x <- str_replace_all(x, "\\b(ocho|octavo|8vo)\\b", " 8 ")
    x <- str_replace_all(x, "\\b(nueve|noveno|9no)\\b", " 9 ")
    x <- str_replace_all(x, "\\b(diez|decimo|10mo)\\b", " 10 ")
    x <- str_replace_all(x, "\\b(once|onceavo|11vo)\\b", " 11 ")
    x <- str_replace_all(x, "\\b(doce|doceavo|12vo)\\b", " 12 ")
    x <- str_replace_all(x, "\\b(trece|13vo)\\b", " 13 ")
    x <- str_replace_all(x, "\\b(catorce|14vo)\\b", " 14 ")
    x <- str_replace_all(x, "\\b(quince|15vo)\\b", " 15 ")
    x <- str_replace_all(x, "\\b(dieciseis|16vo)\\b", " 16 ")
    x <- str_replace_all(x, "\\b(diecisiete|17vo)\\b", " 17 ")
    x <- str_replace_all(x, "\\b(dieciocho|18vo)\\b", " 18 ")
    x <- str_replace_all(x, "\\b(diecinueve|19vo)\\b", " 19 ")
    
    return(x)
  })
  corpus <- tm_map(corpus, replace_numbers)

  # Again white spaces (space in numbers tu attempt to fix mistakes on word boundaries)
  corpus <- tm_map(corpus, content_transformer(stripWhitespace))
  
 # Aquí parte 4 no está sirviendo del todo ( Está al final final del código)
  
  
  return(corpus)
}

# Corpus (still with stopwords) for "Apartamento" 
corpus_train_apto_desc <- clean_corpus(train_apto$description)
corpus_test_apto_desc <- clean_corpus(test_apto$description)


# ------------------------------------------------------------------------------

# See missing % BEFORE imputation
na_report <- tibble(
  variable = names(train_apto),
  train_apto = map_dbl(train_apto, ~ round(mean(is.na(.)) * 100, 2)),
  test_apto  = map_dbl(test_apto,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train_apto))

na_report

# TODO Aquí no he considerado el caso en que descripción y valor no sean igual
# entonces no sé si debería forzar el cambio o dejarlo como está

# Data imputation function
impute_from_description <- function(df, corpus_desc) {
  
  # 1. Convert corpus object into normal text
  desc_text <- sapply(corpus_desc, as.character)
  
  # 2. Generate local copies of variables before imputation
  bathrooms_new         <- df$bathrooms
  rooms_new             <- df$rooms
  bed_rooms_new         <- df$bedrooms
  
  # -------------------------------------------------------------------
  # 3. IMPUTE BATHROOMS — STEP 1: extract explicit numeric patterns
  # Look for expressions like "2 bano"
  found_bath <- str_extract(desc_text, "\\b([0-9]+)\\s*(bano)\\b")
  found_bath_num <- as.numeric(str_extract(found_bath, "[0-9]+"))
  
  # ---- Regla: si > 8 → tomar primer dígito ----
  idx_big_bath <- which(!is.na(found_bath_num) & found_bath_num > 8)
  
  if (length(idx_big_bath) > 0) {
    # Extraer el primer dígito de cada caso
    first_digit <- as.numeric(str_sub(found_bath_num[idx_big_bath], 1, 1))
    found_bath_num[idx_big_bath] <- first_digit
  }
   
  # Count total mentions of "baño"/"bano" variants when no explicit number is found
  bath_count <- str_count(desc_text, "\\b(bano)\\b")
  
  # Impute missing values using explicit numbers | bath_count | 1
  bathrooms_new <- ifelse(!is.na(found_bath_num),
                      found_bath_num, 
                      bathrooms_new)
  # Use the count only for those that are still missing (else 1 by defect)
  bathrooms_new <- ifelse(is.na(bathrooms_new) & bath_count > 0, bath_count, bathrooms_new)
  bathrooms_new <- ifelse(is.na(bathrooms_new), 1, bathrooms_new)

  # -------------------------------------------------------------------
  # 4. IMPUTE ROOMS
  # Look for sentences like "# habitacion"
  found_rooms <- str_extract(desc_text, "\\b([0-9]+)\\s*(habitacion)\\b")
  found_rooms_num <- as.numeric(str_extract(found_rooms, "[0-9]+"))
   
  # 1st, we imputate the data found; not checking if NA or not
  rooms_new <- ifelse(!is.na(found_rooms_num) & found_rooms_num < 11,
                  found_rooms_num, rooms_new)
  
  # If rooms_new = NA, and rooms # was not found, put bedroom # (has no NA)
  rooms_new <- ifelse(is.na(rooms_new) & is.na(found_rooms_num),
                  bed_rooms_new, bed_rooms_new )

  # -------------------------------------------------------------------
  # 5. Imputated values in the apto dataset
  
  df <- df %>%
    mutate(
      bathrooms        = bathrooms_new,
      rooms            = rooms_new,
    )
  
  return(df)
}

# Impute data
train_apto <- impute_from_description(train_apto, corpus_train_apto_desc)
test_apto <- impute_from_description(test_apto, corpus_test_apto_desc)



# Function specifically for the total surface and terrace sizes
impute_terrace_and_total <- function(df, corpus_desc) {
  
  # 1. Convert corpus object into normal text
  desc_text <- sapply(corpus_desc, as.character)
  cat("Longitud df:", nrow(df), " - longitud desc_text:", length(desc_text), "\n")
  
  # 2. Local copies
  rooms_new             <- df$rooms
  bed_rooms_new         <- df$bedrooms
  terraza_new           <- df$terraza
  surface_covered_new   <- df$surface_covered
  
  # --------------------------------------------------------------
  # 3. IMPUTE TERRAZA
  extract_terraza_metros <- function(text) {
    pattern <- "(?:terraza(?:\\s+de)?\\s+([0-9]+)\\s+metros)|(?:([0-9]+)\\s+metros\\s+(?:de\\s+)?terraza)"
    matches <- str_extract_all(text, pattern, simplify = FALSE)[[1]]
    
    if (length(matches) == 0) return(NA)
    
    # Extraer solo el número correcto
    nums <- str_extract(matches, "[0-9]+")
    nums <- as.numeric(nums)
    
    # Tomar la ÚLTIMA coincidencia real (la mayoría de anuncios ponen terraza al final)
    return(tail(nums, 1))
  }
  
  # Vectorizar correctamente
  found_terrace <- sapply(desc_text, extract_terraza_metros)
  found_terrace_num <- as.numeric(found_terrace)
  
  # ---- Regla: si > 999, tomar últimos 3 dígitos ----
  idx_large <- which(!is.na(found_terrace_num) & found_terrace_num > 999)
  if (length(idx_large) > 0) {
    found_terrace_num[idx_large] <- as.numeric(
      str_sub(found_terrace_num[idx_large], -3, -1)
    )
  }
  
  # ---- Regla: si aún > 400 y desproporcionado a nº habitaciones → 2/3 ----
  idx_too_big <- which(
    !is.na(found_terrace_num) &
      found_terrace_num > 400 &
      (is.na(rooms_new) | (found_terrace_num / rooms_new) > 100)
  )
  if (length(idx_too_big) > 0) {
    found_terrace_num[idx_too_big] <- round(found_terrace_num[idx_too_big] * (2/3))
  }
  
  # Imputar terraza cuando se encuentre valor válido
  terraza_new[!is.na(found_terrace_num)] <- found_terrace_num[!is.na(found_terrace_num)]
  
  # Imputar mínimo si aparece palabra "terraza" o "balcon"
  has_terraza_word <- str_detect(desc_text, "terraza")
  has_balcon_word  <- str_detect(desc_text, "balcon")
  
  terraza_new[is.na(terraza_new) & (has_terraza_word | has_balcon_word)] <- 5
  
  # Los NA restantes → 0
  terraza_new[is.na(terraza_new)] <- 0
  
  # --------------------------------------------------------------
  # 4. IMPUTE SIZE (superficie cubierta)
  found_size <- str_extract(
    desc_text,
    "\\b([0-9]+)\\s*metros?(?!\\s*terraza|\\s*de terraza)\\b"
  )
  found_size_num <- as.numeric(str_extract(found_size, "[0-9]+"))
  
  # Debug
  idx_na <- which(is.na(found_size_num))
  if (length(idx_na) > 0) {
    cat("Surface not found in:", length(idx_na), "descriptions\n")
  }
  
  # ---- Regla 1: si > 999 → tomar últimos 3 dígitos,
  #      pero si esos < 30 → tomar primeros 3 dígitos ----
  idx_large <- which(!is.na(found_size_num) & found_size_num > 999)
  
  if (length(idx_large) > 0) {
    
    # últimos 3 dígitos
    last3  <- as.numeric(str_sub(found_size_num[idx_large], -3, -1))
    
    # primeros 3 dígitos
    first3 <- as.numeric(str_sub(found_size_num[idx_large], 1, 3))
    
    # regla: si últimos 3 < 30 → usar primeros 3
    fixed3 <- ifelse(last3 < 30, first3, last3)
    
    # aplicar corrección
    found_size_num[idx_large] <- fixed3
  }
  
  # ---- Regla 2: si > 400 y desproporcionado → 2/3 ----
  idx_too_big <- which(
    !is.na(found_size_num) &
      found_size_num > 400 &
      (is.na(rooms_new) | (found_size_num / rooms_new) > 100)
  )
  
  if (length(idx_too_big) > 0) {
    found_size_num[idx_too_big] <- round(found_size_num[idx_too_big] * (2/3))
  }
  
  # ---- IMPUTACIÓN FINAL: usar solo si mejora la base ----
  idx_update_surface <- which(
    !is.na(found_size_num) &
      (is.na(surface_covered_new) | surface_covered_new < found_size_num)
  )
  
  surface_covered_new[idx_update_surface] <- found_size_num[idx_update_surface]

  
  # --------------------------------------------------------------
  # 5. FINAL: actualizar df
  df <- df %>%
    mutate(
      terraza         = terraza_new,
      surface_covered = surface_covered_new,
      surface_total = case_when(
        is.na(surface_total) & !is.na(surface_covered_new) & !is.na(terraza_new) ~ surface_covered_new + terraza_new,
        is.na(surface_total) & !is.na(surface_covered_new) &  is.na(terraza_new) ~ surface_covered_new,
        is.na(surface_total) &  is.na(surface_covered_new) & !is.na(terraza_new) & terraza_new > 5 ~ 2 * terraza_new,
        !is.na(surface_covered) & !is.na(surface_total) & (surface_total < surface_covered) ~ 
          surface_covered + ifelse(!is.na(terraza) & terraza > 0, terraza, 0),
        TRUE ~ surface_total
      )
    )
  
  return(df)
}


# Impute data
train_apto <- impute_terrace_and_total(train_apto, corpus_train_apto_desc)
test_apto <- impute_terrace_and_total(test_apto, corpus_test_apto_desc)



# See missing % AFTER imputation
na_report <- tibble(
  variable = names(train_apto),
  train_apto = map_dbl(train_apto, ~ round(mean(is.na(.)) * 100, 2)),
  test_apto  = map_dbl(test_apto,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train_apto))

na_report

# ------------------------------------------------------------------------------
# Create new variables using regular expressions and description DTM 

# NEW VARIABLES (key ones)
# Create new variables (parqueaderos, num_piso, etc)
train_apto <- train_apto %>% mutate(parqueaderos = NA_real_, num_piso = NA_real_, remodelado = NA_real_,
                          deposito = NA_real_, balcon = NA_real_)
test_apto <- test_apto %>% mutate(parqueaderos = NA_real_, num_piso = NA_real_, remodelado = NA_real_,
                          deposito = NA_real_, balcon = NA_real_)

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
  # 3. CREATE PARQUEADERO
  # Look for patterns like "2 parqueaderos", "parqueadero 1", or "1 parqueadero"
  found_park <- str_extract(desc_text, "\\b([0-9]+)\\s*parqueadero\\b")
  found_park_num <- as.numeric(str_extract(found_park, "[0-9]+"))
  
  # If no number but the word exists, assume 1
  has_park_word <- str_detect(desc_text, "\\bparqueadero\\b")
  
  # ---- Regla: si > 8 → tomar primer dígito ----
  idx_big_park <- which(!is.na(found_park_num) & found_park_num > 3)
  
  if (length(idx_big_park) > 0) {
    found_park_num[idx_big_park] <- 2
  }
  
  parqueadero_new <- ifelse(!is.na(found_park_num),
                            found_park_num,
                            parqueadero_new)
  parqueadero_new <- ifelse(is.na(parqueadero_new) & has_park_word, 1, parqueadero_new)
  parqueadero_new <- ifelse(is.na(parqueadero_new), 0, parqueadero_new)
  
  # -------------------------------------------------------------------
  # 4. CREATE NUM_PISO
  # Look for "piso 3" or "3 piso"
  found_piso <- str_extract(desc_text, "\\b(piso\\s*[0-9]+)|([0-9]+)\\s*piso\\b")
  found_piso_num <- as.numeric(str_extract(found_piso, "[0-9]+"))
  
  # Check piso a normal number
  ifelse(found_piso_num >70, 
         found_piso_num <- as.numeric(str_sub(found_piso_num,1,1)),
         found_piso_num <- found_piso_num)
  
  # minimum floor assumed if no # present
  num_piso_new <- ifelse(!is.na(found_piso_num), found_piso_num, 1)
  
  # -------------------------------------------------------------------
  # 5. CREATE DEPOSITO
  # If the text mentions "deposito" at least once, assume 1
  has_deposito <- str_detect(desc_text, "deposito")
  deposito_new <- ifelse(has_deposito, 1, 0)
  
  # -------------------------------------------------------------------
  # 6. CREATE REMODELADO
  # If the text mentions "remodelado", assume 1
  has_remodelado <- str_detect(desc_text, "remodelado")
  remodelado_new <- ifelse(has_remodelado, 1, 0)
  
  
  # -------------------------------------------------------------------
  # 7. CREATE BALCON
  found_balcon <- str_extract(desc_text, "\\b([0-9]+)\\s*balcon\\b")
  found_balcon_num <- as.numeric(str_extract(found_balcon, "[0-9]+"))
  
  # If no number but the word exists, assume 1
  has_balcon_word <- str_detect(desc_text, "balcon")
  
  balcon_new <- ifelse(!is.na(found_balcon_num),
                            found_balcon_num,
                            balcon_new)
  balcon_new <- ifelse(is.na(balcon_new) & has_balcon_word, 1, balcon_new)
  balcon_new <- ifelse(is.na(balcon_new), 0, balcon_new)

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

# Fill new variables in DB
train_apto <- create_extra_features(train_apto, corpus_train_apto_desc)
test_apto <- create_extra_features(test_apto, corpus_test_apto_desc)

# See missing % AFTER creation
na_report <- tibble(
  variable = names(train_apto),
  train_apto = map_dbl(train_apto, ~ round(mean(is.na(.)) * 100, 2)),
  test_apto  = map_dbl(test_apto,  ~ round(mean(is.na(.)) * 100, 2))
) %>%
  arrange(desc(train_apto))

na_report

# missing_test <- test_apto %>% filter(is.na(surface_total) | surface_total < 30)
# missing_train <- train_apto %>% filter(is.na(surface_total) | surface_total < 30)

# For surface_total, some values are not being considered for special cases, 
# thus for those in NA or lower than 50 meters we consider another strategy
# (see extra)




# ---------------------------------------------------------------------------

# CLEAR VARIABLES ALREADY USED
clean_already_used_variables <- function(corpus){

  # Delete: metros, parqueadero, bano, habitacion, deposito, balcon, remodelado piso, and all numbers
  delete_words <- content_transformer(function(x) {
    x <- str_replace_all(x, "\\b(metros|parqueadero|terraza|bano|habitacion|deposito|balcon|remodelado|piso)\\b", " ")
    x <- str_replace_all(x, "\\b([0-9]+)\\b", " ")
    return(x)
  })
  corpus <- tm_map(corpus, delete_words)
  
  # Clean white spaces again
  corpus <- tm_map(corpus, content_transformer(stripWhitespace))

  return(corpus)
}

corpus_train_apto_desc <- clean_already_used_variables(corpus_train_apto_desc)
corpus_test_apto_desc <- clean_already_used_variables(corpus_test_apto_desc)


# LEMATIZATION

# spacy_install()
# spacy_download_langmodel("es_core_news_md")
spacy_initialize(model = "es_core_news_md")


# Lematize 1 doc per call
lemmatize_one <- function(text) {
  parsed <- spacy_parse(text, lemma = TRUE)
  paste(parsed$lemma, collapse = " ")
}

train_text_clean <- sapply(corpus_train_apto_desc, as.character)
test_text_clean  <- sapply(corpus_test_apto_desc,  as.character)

train_apto$description <- sapply(train_text_clean, lemmatize_one, USE.NAMES = FALSE)
test_apto$description  <- sapply(test_text_clean,  lemmatize_one, USE.NAMES = FALSE)



# STOPWORDS

# Download stopwords
palabras1 <- stopwords(language = "es", source = "snowball")
palabras2 <- stopwords(language = "es", source = "nltk")

# Take words already used as variables
lista_palabras <- union(palabras1, palabras2)

# Apply to train
train_apto$description <- removeWords(train_apto$description, lista_palabras)
train_apto$description <- stripWhitespace(train_apto$description)
# Apply to test
test_apto$description <- removeWords(test_apto$description, lista_palabras)
test_apto$description <- stripWhitespace(test_apto$description)


# DTM MATRIX 

# n-gramas
train_unig <- tokenize_words(train_apto$description)
train_big  <- tokenize_ngrams(train_apto$description, n = 2)
train_trig <- tokenize_ngrams(train_apto$description, n = 3)

tokens_train <- mapply(function(u, b, t) {
  c(u, b, t)   # <<< NO usar paste()
}, train_unig, train_big, train_trig, SIMPLIFY = FALSE)

test_unig <- tokenize_words(test_apto$description)
test_big  <- tokenize_ngrams(test_apto$description, n = 2)
test_trig <- tokenize_ngrams(test_apto$description, n = 3)

tokens_test <- mapply(function(u, b, t) {
  c(u, b, t)
}, test_unig, test_big, test_trig, SIMPLIFY = FALSE)

# Creat DTM train and test
### TRAIN
it_train <- itoken(tokens_train, progressbar = FALSE)
vocab_train <- create_vocabulary(it_train)
vectorizer_train <- vocab_vectorizer(vocab_train)
dtm_train_desc <- create_dtm(it_train, vectorizer_train)
cat("Dimensiones DTM train original:", paste(dim(dtm_train_desc), collapse = " x "), "\n")

### TEST
it_test <- itoken(tokens_test, progressbar = FALSE)
vocab_test <- create_vocabulary(it_test)
vectorizer_test <- vocab_vectorizer(vocab_test)
dtm_test_desc <- create_dtm(it_test, vectorizer_test)
cat("Dimensiones DTM test original:", paste(dim(dtm_test_desc), collapse = " x "), "\n")

# Take words with 90% sparsity (present in at least 10% of documents)
# Con 0.95 quedan test-train (186-160), con 0.9 (84-79), con 0.8 (38-31)
sparsity <- 0.95

apply_sparsity <- function(dtm, sparsity) {
  term_freq <- colSums(dtm > 0)
  min_docs <- nrow(dtm) * (1 - sparsity)
  keep_terms <- term_freq >= min_docs
  dtm_filtered <- dtm[, keep_terms, drop = FALSE]
  return(dtm_filtered)
}

dtm_train_desc <- apply_sparsity(dtm_train_desc, sparsity)
dtm_test_desc  <- apply_sparsity(dtm_test_desc, sparsity)

cat("DTM train después sparsity:", paste(dim(dtm_train_desc), collapse = " x "), "\n")
cat("DTM test después sparsity:", paste(dim(dtm_test_desc), collapse = " x "), "\n")

# Aggregate the DTM as variables and take off description and title
# Convert DTM to normal dataframe
train_desc_df <- as.data.frame(as.matrix(dtm_train_desc))
test_desc_df  <- as.data.frame(as.matrix(dtm_test_desc))

# Combine collumns
train_apto <- cbind(train_apto, train_desc_df)
test_apto  <- cbind(test_apto,  test_desc_df)


# Take ONLY common variables between both test and data
common_cols <- intersect(names(train_apto), names(test_apto))
train_apto <- train_apto[, common_cols]
test_apto  <- test_apto[, common_cols]


# See missing % AFTER creation
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
test_casa <- test %>% filter(property_type == "Casa")
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

# PASTE _APTO AND _CASA TOGETHER FOR BOTH TRAIN AND TEST
# columnas existentes en cada dataset (trian)
cols_apto <- colnames(train_apto)
cols_casa <- colnames(train_casa)

# columnas faltantes en casa
cols_to_add <- setdiff(cols_apto, cols_casa)

# agregar columnas faltantes con 0
for (col in cols_to_add) {
  train_casa[[col]] <- 0
}

# Same with test
cols_apto_test <- colnames(test_apto)
cols_casa_test <- colnames(test_casa)

cols_to_add_test <- setdiff(cols_apto_test, cols_casa_test)

for (col in cols_to_add_test) {
  test_casa[[col]] <- 0
}

# Merge data bases
train_full_finished <- bind_rows(train_apto, train_casa)
test_full_finished  <- bind_rows(test_apto, test_casa)

# Eliminate description and title
train_full_finished <- train_full_finished %>% select(-title, -description)
test_full_finished <- test_full_finished %>% select(-title, -description)

# Exportar
write.csv(train_full_finished,
          file = "data/data_train_text_finished.csv",
          row.names = FALSE,
          fileEncoding = "UTF-8")

write.csv(test_full_finished,
          file = "data/data_test_text_finished.csv",
          row.names = FALSE,
          fileEncoding = "UTF-8")



 # FOR AFTER MERGING WITH GEO-DATA

# Small process to take repeated apartments and missing descriptions
# To take repeated collumns in train
df_clean <- df %>% 
  distinct(lat, lon, description, .keep_all = TRUE)

# Clear those without description
df <- df %>% 
  filter(!is.na(description) & str_trim(description) != "")










# =====================================================================================
# EXTRA
# =====================================================================================
# 4. Search for big numbers in words ("ciento cuarenta y 2")
# Number dictionary
num_words <- list(
  units = c(
    "cero"=0,"uno"=1,"una"=1,"dos"=2,"tres"=3,"cuatro"=4,"cinco"=5,"seis"=6,
    "siete"=7,"ocho"=8,"nueve"=9,"diez"=10,"once"=11,"doce"=12,"trece"=13,
    "catorce"=14,"quince"=15),
  
  tens = c(
    "veinte"=20,"treinta"=30,"cuarenta"=40,"cincuenta"=50,"sesenta"=60,
    "setenta"=70,"ochenta"=80,"noventa"=90),
  
  teens = c(
    "dieciseis"=16,"diecisiete"=17,"dieciocho"=18,"diecinueve"=19,
    "veintiuno"=21,"veintidos"=22,"veintitres"=23),
  
  hundreds = c(
    "cien"=100,"ciento"=100,"doscientos"=200,"trescientos"=300,"cuatrocientos"=400,
    "quinientos"=500,"seiscientos"=600,"setecientos"=700,
    "ochocientos"=800,"novecientos"=900)
)

convert_number_phrase <- function(phrase) {
  words <- str_split(phrase, "\\s+")[[1]]
  total <- 0
  current <- 0
  
  for (w in words) {
    
    if (str_detect(w, "^[0-9]+$")) { 
      current <- current + as.numeric(w) 
      next 
    }
    
    if (w %in% names(num_words$hundreds)) {
      current <- current + num_words$hundreds[[w]]; next }
    if (w %in% names(num_words$teens)) {
      current <- current + num_words$teens[[w]]; next }
    if (w %in% names(num_words$tens)) {
      current <- current + num_words$tens[[w]]; next }
    if (w %in% names(num_words$units)) {
      current <- current + num_words$units[[w]]; next }
  }
  total + current
}

# Only transform phrases with "metros" before
convert_metros <- content_transformer(function(x){
  
  patron <- paste0(
    "(?<frase>(?:\\w+\\s+){1,4}\\w+)(?=\\s+metros\\b)"
  )
  
  str_replace_all(x, patron, function(m){
    
    frase <- str_trim(m)
    
    # If its NA, no process
    if (is.na(frase)) return(frase)
    
    # Ignore if its already a number ("234")
    if (str_detect(frase, "^\\d+$")) return(frase)
    
    palabras <- str_split(frase, "\\s+")[[1]]
    
    # If word not in dictionary, no changes applied
    if (!any(palabras %in% unlist(lapply(num_words, names))) &&
        !any(str_detect(palabras, "^[0-9]+$"))) {
      return(frase)
    }
    
    num <- convert_number_phrase(frase)
    return(as.character(num))
  })
})

corpus <- tm_map(corpus, convert_metros)" # esto funciona a medias

