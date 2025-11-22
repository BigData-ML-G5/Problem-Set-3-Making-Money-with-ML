# =======================================================
# PS3 — Chapinero Prices
# XGBoost + Spatial CV (tidymodels)
# =======================================================

# ------------------------
# 1) Cargar librerías
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,   # recipes, parsnip, workflows, tune, yardstick
    sf,           # st_as_sf
    spatialsample,# spatial_block_cv
    xgboost       # motor XGBoost
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

# Igual que en tus otros scripts: eliminar property_type si existe
if ("property_type" %in% names(train)) {
  train <- train %>% dplyr::select(-property_type)
}
if ("property_type" %in% names(test)) {
  test  <- test  %>% dplyr::select(-property_type)
}

# Chequeos básicos
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))  # para CV espacial

# Asegurar que price sea numérico
train$price <- as.numeric(train$price)

# ------------------------
# 3) Folds de validación cruzada ESPACIAL
# ------------------------
set.seed(2025)

train_sf <- sf::st_as_sf(
  train,
  coords = c("lon", "lat"),
  crs    = 4326,
  remove = FALSE
)

block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# ------------------------
# 4) Receta para XGBoost
#    - Imputación numérica y categórica
#    - Dummies para factores (XGBoost necesita matriz numérica)
# ------------------------
rec_xgb <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%  # no usar como predictor
  # quitar geometry si quedó del sf
  step_rm(geometry, skip = TRUE) %>%
  # manejar niveles nuevos en test
  step_novel(all_nominal_predictors()) %>%
  # imputaciones
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # crear dummies para categóricas
  step_dummy(all_nominal_predictors())

# ------------------------
# 5) Especificación del modelo XGBoost + hiperparámetros
# ------------------------
# Aproximamos p con los predictores crudos (antes de dummies)
p <- length(setdiff(names(train), c("price", "property_id")))

xgb_spec <- boost_tree(
  trees        = 650,                 # similar a tu código viejo
  mtry         = floor(sqrt(p)),      # regla clásica (no tuneamos mtry)
  min_n        = tune(),              # ~ min_child_weight
  tree_depth   = tune(),              # max_depth
  learn_rate   = tune(),              # eta
  sample_size  = tune(),              # subsample
  loss_reduction = 0                  # gamma = 0 (fijo)
) %>%
  set_mode("regression") %>%
  set_engine(
    "xgboost",
    objective   = "reg:squarederror",
    eval_metric = "rmse"
  )

wf_xgb <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rec_xgb)

# ------------------------
# 6) Grid de hiperparámetros (tuning)
# ------------------------
xgb_grid <- crossing(
  min_n       = c(5, 20),          # nodos pequeños vs más grandes
  tree_depth  = c(4, 7, 10),       # poco profundo, medio, profundo
  learn_rate  = c(0.03, 0.06),     # similar a tu 0.06
  sample_size = c(0.7, 0.9)        # subsample
)
print(xgb_grid)

# ------------------------
# 7) Tuning con CV espacial usando MAE
# ------------------------
set.seed(2025)
xgb_tuned <- tune_grid(
  wf_xgb,
  resamples = block_folds,
  grid      = xgb_grid,
  metrics   = metric_set(mae),
  control   = control_grid(save_pred = TRUE)
)

xgb_metrics <- collect_metrics(xgb_tuned)
print(xgb_metrics)

# Guardar resultados completos de tuning
readr::write_csv(xgb_metrics, "XGB_tm_spatialcv5_tuning_mae.csv")

# Mejor combinación según MAE
xgb_best <- xgb_metrics %>%
  dplyr::filter(.metric == "mae") %>%
  dplyr::arrange(mean) %>%
  dplyr::slice(1)

print(xgb_best)
readr::write_csv(xgb_best, "XGB_tm_spatialcv5_best_mae.csv")

# ------------------------
# 8) Entrenar modelo final en TODO el train con hiperparámetros óptimos
# ------------------------
best_params <- xgb_best %>%
  dplyr::select(min_n, tree_depth, learn_rate, sample_size)

wf_xgb_final <- finalize_workflow(
  wf_xgb,
  best_params
)

set.seed(2025)
fit_xgb <- fit(wf_xgb_final, data = train)

cat("XGBoost final - hiperparámetros:\n")
print(best_params)
cat("Árboles (trees): 650\n")
cat("mtry (fijo):", floor(sqrt(p)), "\n")
cat("loss_reduction (gamma): 0\n")

# ------------------------
# 9) Variables efectivamente usadas (después de la receta)
# ------------------------
prep_rec <- prep(rec_xgb, training = train)
baked_train <- bake(prep_rec, new_data = train)

vars_used <- setdiff(names(baked_train), "price")
write_csv(tibble(var = vars_used), "XGB_tm_vars_used.csv")

# ------------------------
# 10) Predicciones sobre TEST y archivo para Kaggle
# ------------------------
pred_test <- predict(fit_xgb, new_data = test)$.pred

# Redondeo hacia abajo al millón más cercano
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = as.numeric(pred_test_round)
)

print(head(submission))

model_label <- "XGB_tm"
cv_label    <- "spatialcv5"

fname <- paste0(model_label, "_", cv_label, ".csv")
write.csv(submission, fname, row.names = FALSE)
cat("Archivo de submission guardado como:", fname, "\n")

# ------------------------
# End of script
# ------------------------
