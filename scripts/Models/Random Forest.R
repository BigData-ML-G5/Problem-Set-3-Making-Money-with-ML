# =======================================================
# PS3 — Chapinero Prices
# Random Forest (ranger) + Spatial CV (tidymodels)
# =======================================================

# ------------------------
# 1) Cargar librerías
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,
    sf,
    spatialsample,
    ranger,
    doParallel         # <--- para paralelizar tune_grid
  )
})

tidymodels_prefer()

# nº de núcleos a usar (deja 1 libre)
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1
message("Usando ", n_cores, " núcleos")

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

# ------------------------
# 3) Quitar property_type (igual que en SL3 / SL viejo)
# ------------------------
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
stopifnot(all(c("lon", "lat") %in% names(train)))

train$price <- as.numeric(train$price)

# ------------------------
# 4) Folds espaciales (spatial_block_cv)
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
# 5) Receta ligera (sin dummies)
#    RF maneja bien numéricas + factores
# ------------------------
rec_rf <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

# ------------------------
# 6) Especificación del modelo RF
#    Usando varios hilos dentro de ranger
# ------------------------
rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000          # nº de árboles
) %>%
  set_engine("ranger", num.threads = n_cores) %>%  # <--- multi-thread
  set_mode("regression")

wf_rf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rec_rf)

# ------------------------
# 7) Grid de hiperparámetros
# ------------------------
p <- length(setdiff(names(train), c("price", "property_id")))

mtry_vals  <- unique(pmax(1, c(floor(sqrt(p)), floor(p / 3), floor(p / 2))))
min_n_vals <- c(5, 20, 50)

rf_grid <- tidyr::crossing(
  mtry  = mtry_vals,
  min_n = min_n_vals
)

print(rf_grid)

# ------------------------
# 8) Spatial CV + tuning (MAE) en paralelo
# ------------------------

# Crear cluster para paralelizar por resample
cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)

control_rf <- control_grid(
  parallel_over = "resamples",  # folds en paralelo
  verbose       = TRUE
)

set.seed(2025)
rf_tuned <- tune_grid(
  wf_rf,
  resamples = block_folds,
  grid      = rf_grid,
  metrics   = metric_set(mae),
  control   = control_rf
)



rf_metrics <- collect_metrics(rf_tuned)
print(rf_metrics)

# Guardar resultados completos de tuning
readr::write_csv(rf_metrics, "RF_tm_spatialcv5_tuning_mae.csv")

# Mejor combinación según MAE
rf_best <- rf_metrics %>%
  dplyr::filter(.metric == "mae") %>%
  dplyr::arrange(mean) %>%
  dplyr::slice(1)

print(rf_best)
readr::write_csv(rf_best, "RF_tm_spatialcv5_best_mae.csv")

# ------------------------
# 9) Entrenar modelo final con hiperparámetros óptimos
# ------------------------
best_params <- rf_best %>% dplyr::select(mtry, min_n)
print(best_params)

wf_rf_final <- finalize_workflow(
  wf_rf,
  best_params
)

set.seed(2025)
fit_rf <- fit(wf_rf_final, data = train)

# ------------------------
# 11) Predicciones para Kaggle
# ------------------------
pred_test <- predict(fit_rf, new_data = test)$.pred
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = as.numeric(pred_test_round)
)

# ------------------------
# 12) Guardar archivo Kaggle con hiperparámetros en el nombre
# ------------------------

mtry_best  <- dplyr::pull(best_params, mtry)
min_n_best <- dplyr::pull(best_params, min_n)

file_name <- sprintf(
  "submission_rf_mtry%s_minN%s.csv",
  mtry_best,
  min_n_best
)

readr::write_csv(submission, file_name)