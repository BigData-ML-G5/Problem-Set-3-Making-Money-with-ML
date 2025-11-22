# =======================================================
# PS3 — Chapinero Prices
# XGBoost + Spatial CV (tidymodels)
# =======================================================

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,
    sf,
    spatialsample,
    xgboost,
    doParallel
  )
})

tidymodels_prefer()

# ------------------------
# 1) Cargar datos
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_train_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_test_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

if ("property_type" %in% names(train)) train <- train %>% select(-property_type)
if ("property_type" %in% names(test))  test  <- test  %>% select(-property_type)

stopifnot(all(c("lon","lat") %in% names(train)))

train$price <- as.numeric(train$price)

# ------------------------
# 2) Spatial CV folds
# ------------------------
set.seed(2025)
train_sf <- st_as_sf(train, coords=c("lon","lat"), crs=4326, remove=FALSE)
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# ------------------------
# 3) Receta (dummies necesarias para XGBoost)
# ------------------------
rec_xgb <- recipe(price ~ ., data=train) %>%
  update_role(property_id, new_role="id") %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

# ------------------------
# 4) Modelo XGBoost
# ------------------------
p <- length(setdiff(names(train), c("price", "property_id")))

xgb_spec <- boost_tree(
  trees        = 650,
  mtry         = floor(sqrt(p)),
  min_n        = tune(),
  tree_depth   = tune(),
  learn_rate   = tune(),
  sample_size  = tune(),
  loss_reduction = 0
) %>%
  set_mode("regression") %>%
  set_engine("xgboost", objective="reg:squarederror")

wf_xgb <- workflow() %>% add_model(xgb_spec) %>% add_recipe(rec_xgb)

# ------------------------
# 5) Grid
# ------------------------
xgb_grid <- tidyr::crossing(
  min_n       = c(5, 20),
  tree_depth  = c(4, 7, 10),
  learn_rate  = c(0.03, 0.06),
  sample_size = c(0.7, 0.9)
)

# ------------------------
# 6) Tuning en paralelo (más rápido)
# ------------------------

n_cores <- parallel::detectCores() - 1
cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)

control_xgb <- control_grid(
  parallel_over = "resamples",
  verbose = TRUE
)

set.seed(2025)
xgb_tuned <- tune_grid(
  wf_xgb,
  resamples = block_folds,
  grid      = xgb_grid,
  metrics   = metric_set(mae),
  control   = control_xgb
)

parallel::stopCluster(cl)
foreach::registerDoSEQ()

xgb_metrics <- collect_metrics(xgb_tuned)
print(xgb_metrics)

# ------------------------
# 7) Mejor modelo
# ------------------------
xgb_best <- xgb_metrics %>%
  filter(.metric=="mae") %>%
  arrange(mean) %>% slice(1)

print(xgb_best)

best_params <- xgb_best %>% select(min_n, tree_depth, learn_rate, sample_size)

# ------------------------
# 8) Entrenar modelo final
# ------------------------
wf_xgb_final <- finalize_workflow(wf_xgb, best_params)

fit_xgb <- fit(wf_xgb_final, data=train)

# ------------------------
# 9) Predicciones test
# ------------------------
pred_test <- predict(fit_xgb, new_data=test)$.pred
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = pred_test_round
)

# ------------------------
# 10) Guardar archivo Kaggle con hiperparámetros
# ------------------------
fname <- sprintf(
  "XGB_tm_minn%s_depth%s_lr%s_ss%s.csv",
  best_params$min_n,
  best_params$tree_depth,
  best_params$learn_rate,
  best_params$sample_size
)

write.csv(submission, fname, row.names=FALSE)