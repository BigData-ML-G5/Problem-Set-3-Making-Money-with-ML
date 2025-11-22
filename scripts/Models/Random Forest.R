# =======================================================
# PS3 — Chapinero Prices
# Random Forest (ranger) + Spatial CV (tidymodels)
# =======================================================

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,
    sf,
    spatialsample,
    ranger
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

# ------------------------
# Quitar property_type (igual que en SL3)
# ------------------------
if ("property_type" %in% names(train)) {
  train <- train %>% select(-property_type)
}
if ("property_type" %in% names(test)) {
  test  <- test  %>% select(-property_type)
}

# Chequeos básicos
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))

train$price <- as.numeric(train$price)

# ------------------------
# 3) Folds espaciales
# ------------------------
set.seed(2025)

train_sf <- sf::st_as_sf(
  train,
  coords = c("lon","lat"),
  crs = 4326,
  remove = FALSE
)

block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# ------------------------
# 4) Receta ligera (sin dummies)
# ------------------------
rec_rf <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%
  step_rm(geometry, skip = TRUE) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

# ------------------------
# 5) Modelo
# ------------------------
rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

wf_rf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rec_rf)

# ------------------------
# 6) Grid de hiperparámetros
# ------------------------
p <- length(setdiff(names(train), c("price", "property_id")))

mtry_vals  <- unique(pmax(1, c(floor(sqrt(p)), floor(p/3), floor(p/2))))
min_n_vals <- c(5, 20, 50)

rf_grid <- crossing(
  mtry  = mtry_vals,
  min_n = min_n_vals
)

# ------------------------
# 7) Spatial CV + tuning
# ------------------------
set.seed(2025)
rf_tuned <- tune_grid(
  wf_rf,
  resamples = block_folds,
  grid      = rf_grid,
  metrics   = metric_set(mae),
  control   = control_grid(save_pred = TRUE)
)

rf_metrics <- collect_metrics(rf_tuned)
readr::write_csv(rf_metrics, "RF_tm_spatialcv5_tuning_mae.csv")

rf_best <- rf_metrics %>%
  filter(.metric == "mae") %>%
  arrange(mean) %>%
  slice(1)

readr::write_csv(rf_best, "RF_tm_spatialcv5_best_mae.csv")

# ------------------------
# 8) Entrenar modelo final
# ------------------------
best_params <- rf_best %>% select(mtry, min_n)

wf_rf_final <- finalize_workflow(
  wf_rf,
  best_params
)

fit_rf <- fit(wf_rf_final, data = train)

prep_rec <- prep(rec_rf, training = train)
baked_train <- bake(prep_rec, new_data = train)

vars_used <- setdiff(names(baked_train), "price")
write_csv(tibble(var = vars_used), "RF_tm_vars_used.csv")

# ------------------------
# 9) Predicciones para Kaggle
# ------------------------
pred_test <- predict(fit_rf, new_data = test)$.pred
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = pred_test_round
)

write.csv(submission, "RF_tm_spatialcv5.csv", row.names = FALSE)
cat("Archivo creado: RF_tm_spatialcv5.csv\n")

# ------------------------
# FIN
# ------------------------
