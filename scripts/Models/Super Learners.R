# =======================================================
# PS3 — Chapinero Prices
# SuperLearner + Spatial CV
# =======================================================

# ------------------------
# 1) Cargar librerías
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,     # recipes, etc.
    sf,             # st_as_sf
    spatialsample,  # spatial_block_cv
    rsample,        # analysis()/assessment()
    SuperLearner,   # SuperLearner principal
    glmnet,         # para SL.glmnet
    ranger,         # para SL.ranger
    earth           # para SL.earth (MARS)
    # si quieres luego: xgboost para SL.xgboost
  )
})

tidymodels_prefer()

# ------------------------
# 2) Cargar datos
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_train_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_test_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

# Chequeos básicos
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))  # para CV espacial

# Aseguramos que price sea numérico
train$price <- as.numeric(train$price)

# ------------------------
# 3) Folds de validación cruzada ESPACIAL (para SuperLearner)
# ------------------------
set.seed(2025)

# Construimos objeto sf con lon/lat
train_sf <- sf::st_as_sf(
  train,
  coords = c("lon", "lat"),
  crs    = 4326,
  remove = FALSE
)

# Bloques espaciales (5 folds, igual que en tus otros scripts)
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# SuperLearner necesita, para cada fold, los índices de VALIDACIÓN (validRows)
validRows <- lapply(block_folds$splits, function(s) {
  which(rownames(train) %in% rownames(rsample::assessment(s)))
})

# ------------------------
# 4) Receta de preprocesamiento (igual filosofía que LM/NN)
# ------------------------
rec_sl <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%  # no usar como predictor
  # manejar niveles nuevos en test para variables categóricas
  step_novel(all_nominal_predictors()) %>%
  # imputación
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # dummies para categóricas
  step_dummy(all_nominal_predictors()) %>%
  # normalizar predictores numéricos
  step_normalize(all_numeric_predictors(), -all_outcomes())

# Preparamos receta con train
prep_sl <- prep(rec_sl, training = train)

# "Horneamos" train y test con la misma receta
train_baked <- bake(prep_sl, new_data = train)
test_baked  <- bake(prep_sl, new_data = test)

# ------------------------
# 5) Construir X, y para SuperLearner
# ------------------------
# Columnas predictoras: todo menos price y property_id
x_cols <- names(train_baked)
x_cols <- setdiff(x_cols, c("price", "property_id"))

X_train <- as.data.frame(train_baked[, x_cols, drop = FALSE])
y_train <- as.numeric(train_baked$price)

X_test  <- as.data.frame(test_baked[, x_cols, drop = FALSE])

cat("Dimensiones X_train:", dim(X_train), "\n")
cat("Dimensiones X_test :", dim(X_test), "\n")

# ------------------------
# 6) Definir la librería de modelos base (learners)
# ------------------------
# Buen conjunto: penalizado, RF, MARS, árbol, media
sl_lib <- c(
  "SL.glmnet",  # elastic net / lasso / ridge
  "SL.ranger",  # random forest rápido
  "SL.earth",   # MARS (no lineal flexible)
  "SL.rpart",   # árbol de decisión
  "SL.mean"     # benchmark simple
)

# ------------------------
# 7) Entrenar SuperLearner con CV ESPACIAL
# ------------------------
set.seed(2025)

fit_sl <- SuperLearner(
  Y          = y_train,
  X          = X_train,
  SL.library = sl_lib,
  family     = gaussian(),          # regresión
  method     = "method.NNLS",       # combinación no-negativa óptima (MSE)
  cvControl  = list(
    V         = length(validRows),  # número de folds
    validRows = validRows           # aquí fijamos CV ESPACIAL
  )
)

print(fit_sl)

# Pesos de cada learner
sl_weights <- fit_sl$coef
print(sl_weights)

# Guardar pesos en csv
weights_df <- tibble(
  learner = names(sl_weights),
  weight  = as.numeric(sl_weights)
)
readr::write_csv(weights_df, "SL_spatial_weights.csv")

# ------------------------
# 8) Desempeño in-sample aproximado (MAE sobre train)
# ------------------------
pred_train_sl <- predict(fit_sl, newdata = X_train)$pred
mae_train_sl  <- mean(abs(pred_train_sl - y_train), na.rm = TRUE)
cat("MAE (aprox, train):", mae_train_sl, "\n")

# ------------------------
# 9) Predicciones sobre TEST y archivo para Kaggle
# ------------------------
pred_test_sl <- predict(fit_sl, newdata = X_test)$pred
pred_test_sl <- as.numeric(pred_test_sl)

# Redondeo hacia abajo al millón más cercano
pred_test_sl_round <- floor(pred_test_sl / 1e6) * 1e6

submission_sl <- tibble(
  property_id = test$property_id,
  price       = pred_test_sl_round
)

print(head(submission_sl))

# Nombre del archivo
model_label <- "SuperLearner"
cv_label    <- "spatialcv5"

fname_sl <- paste0(model_label, "_", cv_label, ".csv")
write.csv(submission_sl, fname_sl, row.names = FALSE)
cat("Archivo de submission guardado como:", fname_sl, "\n")

# ------------------------
# 10) Guardar lista de variables usadas
# ------------------------
vars_used_sl <- x_cols
write.csv(
  data.frame(var = vars_used_sl),
  "SuperLearner_vars_used.csv",
  row.names = FALSE
)

# ------------------------
# End of script
# ------------------------
