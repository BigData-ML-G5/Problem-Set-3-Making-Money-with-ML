# =======================================================
# PS3 — Chapinero Prices
# Linear Regression + Spatial CV (tidymodels)
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,   # recipes, parsnip, workflows, rsample, yardstick, tune
    sf,           # st_as_sf
    spatialsample # spatial_block_cv
  )
})

tidymodels_prefer()  # evitar conflictos de funciones

# ------------------------
# 2) Load data
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
# 3) Definir folds de validación cruzada ESPACIAL
# ------------------------
set.seed(2025)

# Creamos objeto sf usando lon/lat (conservando columnas originales)
train_sf <- sf::st_as_sf(
  train,
  coords = c("lon", "lat"),
  crs    = 4326,
  remove = FALSE
)

# Folds espaciales por bloques
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# ------------------------
# 4) Receta: imputación + dummies
# ------------------------
rec_lm <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>% # no usar como predictor
  # manejar niveles nuevos en test para variables categóricas
  step_novel(all_nominal_predictors()) %>%
  # imputación:
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # crear dummies
  step_dummy(all_nominal_predictors())

# ------------------------
# 5) Especificación del modelo: LM simple
# ------------------------
lm_spec <- linear_reg() %>%
  set_engine("lm")

wf_lm <- workflow() %>%
  add_model(lm_spec) %>%
  add_recipe(rec_lm)

# ------------------------
# 6) Validación cruzada espacial con MAE
# ------------------------
set.seed(2025)
res_lm <- fit_resamples(
  wf_lm,
  resamples = block_folds,
  metrics   = metric_set(mae),
  control   = control_resamples(save_pred = TRUE)
)

# Ver MAE promedio
lm_mae <- collect_metrics(res_lm) %>% 
  filter(.metric == "mae")
print(lm_mae)
readr::write_csv(lm_mae, "LM_tm_spatialcv5_mae.csv")

# ------------------------
# 7) Entrenar modelo final en TODO el train
# ------------------------
fit_lm <- fit(wf_lm, data = train)

# (Opcional) guardar variables efectivamente usadas luego de la receta
prep_rec <- prep(rec_lm, training = train)
baked_train <- bake(prep_rec, new_data = train)
vars_used <- setdiff(names(baked_train), "price")
readr::write_csv(tibble(var = vars_used), "LM_tm_vars_used.csv")

# ------------------------
# 8) Predicciones sobre TEST y archivo para Kaggle
# ------------------------
pred_test <- predict(fit_lm, new_data = test)$.pred

# Redondeo hacia abajo al millón más cercano
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = as.numeric(pred_test_round)  # en COP
)

print(head(submission))

# ------------------------
# 9) Nombre del archivo
# ------------------------
model_label <- "LM_tm"       # LM con tidymodels
cv_label    <- "spatialcv5"

fname <- paste0(model_label, "_", cv_label, ".csv")
write.csv(submission, fname, row.names = FALSE)
cat("Archivo de submission guardado como:", fname, "\n")

# ------------------------
# End of script
# ------------------------

