# =======================================================
# PS3 — Chapinero Prices
# Elastic Net + Spatial CV (caret)
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse, dplyr, tidyr, readr, ggplot2, forcats,
    caret,
    glmnet,        # motor Elastic Net
    sf,            # st_as_sf
    spatialsample, # spatial_block_cv
    rsample        # analysis()/assessment()
  )
})

# ------------------------
# 2) Load data
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_train_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

train <- train %>% select(-property_type)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_test_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

test <- test %>% select(-property_type)

# Chequeos básicos
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))  # para CV espacial

# ------------------------
# 3) Target y selección de columnas
# ------------------------
# Aseguramos que price sea numérico
train$price <- as.numeric(train$price)

# Columnas que NO queremos como predictores
cols_drop    <- c("property_id", "price")
x_cols_train <- setdiff(names(train), cols_drop)
x_cols_test  <- setdiff(names(test),  "property_id")

common_cols  <- intersect(x_cols_train, x_cols_test)
if (length(common_cols) == 0) stop("No common predictors between train and test. Check column names.")

x_cols <- common_cols

# Data final para modelar
train_df <- train %>% dplyr::select(all_of(c("price", x_cols)))
test_df  <- test  %>% dplyr::select(all_of(c("property_id", x_cols)))

# ------------------------
# 4) Imputación simple de NAs y tipos (como en LM)
# ------------------------
impute_mode <- function(v) {
  tb <- table(v, useNA = "no")
  names(tb)[which.max(tb)]
}

for (nm in names(train_df)) {
  if (nm == "price") next
  
  if (is.numeric(train_df[[nm]])) {
    # Numéricos: mediana
    med <- median(train_df[[nm]], na.rm = TRUE)
    train_df[[nm]][is.na(train_df[[nm]])] <- med
    if (nm %in% names(test_df)) {
      test_df[[nm]][is.na(test_df[[nm]])] <- med
    }
  } else {
    # Categóricos/character: factor + moda + alineación de niveles
    train_df[[nm]] <- as.factor(train_df[[nm]])
    mode_val <- impute_mode(train_df[[nm]])
    train_df[[nm]][is.na(train_df[[nm]])] <- mode_val
    
    if (nm %in% names(test_df)) {
      test_df[[nm]] <- as.factor(test_df[[nm]])
      test_df[[nm]] <- forcats::fct_explicit_na(test_df[[nm]], na_level = mode_val)
      test_df[[nm]] <- forcats::fct_other(
        test_df[[nm]],
        keep        = levels(train_df[[nm]]),
        other_level = mode_val
      )
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
      test_df[[nm]][is.na(test_df[[nm]])] <- mode_val
    }
  }
}

# ------------------------
# 5) Validación cruzada ESPACIAL (5 folds) + MAE
# ------------------------
set.seed(2025)

# Convertir train_df a objeto sf usando lon/lat (para definir bloques)
train_sf <- sf::st_as_sf(
  train_df,
  coords = c("lon", "lat"),  # columnas ya imputadas
  crs    = 4326
)

# Crear folds espaciales (bloques)
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# Pasar de rsample a índices para caret
index_list <- lapply(block_folds$splits, function(s) {
  which(rownames(train_df) %in% rownames(rsample::analysis(s)))
})

indexOut_list <- lapply(block_folds$splits, function(s) {
  which(rownames(train_df) %in% rownames(rsample::assessment(s)))
})

maeSummary <- function(data, lev = NULL, model = NULL) {
  c(MAE = caret::MAE(pred = data$pred, obs = data$obs))
}

ctrl_spatial_en <- trainControl(
  method          = "cv",
  number          = length(index_list),  # v = 5
  summaryFunction = maeSummary,
  index           = index_list,
  indexOut        = indexOut_list,
  verboseIter     = TRUE
)

# ------------------------
# 6) Grid de hiperparámetros (alpha, lambda) para Elastic Net
# ------------------------
tune_grid_en <- expand.grid(
  alpha  = seq(0.2, 1.0, by = 0.2),            # 0.2, 0.4, ..., 1.0
  lambda = seq(0.0005, 0.01, length.out = 10)  # ajusta si quieres
)

# ------------------------
# 7) Entrenar Elastic Net (glmnet) con CV espacial
# ------------------------
set.seed(2025)
model_en <- caret::train(
  price ~ .,
  data       = train_df,
  method     = "glmnet",
  trControl  = ctrl_spatial_en,
  tuneGrid   = tune_grid_en,
  metric     = "MAE",
  preProcess = c("center", "scale"),   # estandarizar X para glmnet
  family     = "gaussian"              # regresión
)

print(model_en)
print(model_en$bestTune)

# Guardar resultados de CV (MAE por combinación alpha-lambda)
cv_results_en <- model_en$results
readr::write_csv(cv_results_en, "EN_caret_spatialcv5_tuning_mae.csv")

# MAE promedio del mejor modelo
best_row_en <- cv_results_en %>%
  dplyr::filter(
    alpha  == model_en$bestTune$alpha,
    lambda == model_en$bestTune$lambda
  )
print(best_row_en)
readr::write_csv(best_row_en, "EN_caret_spatialcv5_best_mae.csv")

# ------------------------
# 8) Predicciones sobre TEST y archivo para Kaggle
# ------------------------
pred_price_en <- predict(
  model_en,
  newdata = test_df %>% dplyr::select(all_of(x_cols))
)

if (any(is.na(pred_price_en))) {
  warning("Hay NAs en las predicciones; revisa la imputación.")
}

# Redondeo hacia abajo al millón más cercano
pred_price_en_round <- floor(pred_price_en / 1e6) * 1e6

submission_en <- tibble(
  property_id = test_df$property_id,
  price       = as.numeric(pred_price_en_round)  # en COP
)

print(head(submission_en))

# ------------------------
# 9) Nombre del archivo según hiperparámetros óptimos
# ------------------------
lambda_val <- model_en$bestTune$lambda
alpha_val  <- model_en$bestTune$alpha

lambda_str <- gsub("\\.", "_", format(round(lambda_val, 6), scientific = FALSE))
alpha_str  <- gsub("\\.", "_", format(round(alpha_val, 3), scientific = FALSE))

model_label <- "EN_caret"
cv_label    <- "spatialcv5"

fname_en <- paste0(
  model_label, "_", cv_label,
  "_lambda_", lambda_str,
  "_alpha_", alpha_str,
  ".csv"
)

write.csv(submission_en, fname_en, row.names = FALSE)
cat("Archivo de submission guardado como:", fname_en, "\n")

# ------------------------
# 10) Guardar lista de variables usadas
# ------------------------
vars_used_en <- x_cols
write.csv(data.frame(var = vars_used_en), "EN_caret_vars_used.csv", row.names = FALSE)

# ------------------------
# End of script
# ------------------------
