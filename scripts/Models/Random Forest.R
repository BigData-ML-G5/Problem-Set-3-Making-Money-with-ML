# =======================================================
# PS3 — Chapinero Prices
# Random Forest 
# =======================================================

# -------------------------------------------------------
# 1) Load libraries
# -------------------------------------------------------
# pacman automatically installs missing packages and loads them
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,      # Data wrangling
    tidymodels,     # Modeling framework (recipes, parsnip, workflows)
    sf,             # Spatial objects (simple features)
    spatialsample,  # spatial_block_cv for spatial CV
    ranger,         # Fast implementation of Random Forest
    doParallel      # Parallel backend for tune_grid
  )
})

tidymodels_prefer()   # Avoid function conflicts between packages

# Detect number of cores and leave one free
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1
message("Using ", n_cores, " CPU cores for parallel processing")

# -------------------------------------------------------
# 2) Load final TRAIN and TEST datasets
# -------------------------------------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

# Remove property_type if present (not used as predictor)
if ("property_type" %in% names(train)) train <- train %>% select(-property_type)
if ("property_type" %in% names(test))  test  <- test  %>% select(-property_type)

# Basic integrity checks
stopifnot("property_id" %in% names(train))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))  # Required for spatial CV

# Ensure price is numeric
train$price <- as.numeric(train$price)

# -------------------------------------------------------
# 3) Define spatial cross-validation folds (5-fold block CV)
# -------------------------------------------------------
set.seed(2025)

# Convert dataset to sf using lon/lat, keeping all original columns
train_sf <- sf::st_as_sf(
  train,
  coords = c("lon","lat"),
  crs    = 4326,
  remove = FALSE
)

# Create spatially separated folds to avoid leakage via spatial autocorrelation
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# -------------------------------------------------------
# 4) Preprocessing recipe
# -------------------------------------------------------
# Random Forest handles numeric + factor predictors directly,
# so we do not create dummy variables.
# Only simple imputation + handling novel categories is required.

rec_rf <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%       # Exclude from predictors
  step_novel(all_nominal_predictors()) %>%            # Handle unseen categories
  step_impute_median(all_numeric_predictors()) %>%    # Median imputation
  step_impute_mode(all_nominal_predictors())          # Mode imputation

# -------------------------------------------------------
# 5) Random Forest model specification (ranger)
# -------------------------------------------------------
rf_spec <- rand_forest(
  mtry  = tune(),   # Number of variables tried at each split
  min_n = tune(),   # Minimum samples per leaf
  trees = 1000      # Number of trees (fixed)
) %>%
  set_engine("ranger", num.threads = n_cores) %>%      # Parallel engine
  set_mode("regression")

# Combine model + recipe into a workflow
wf_rf <- workflow() %>%
  add_recipe(rec_rf) %>%
  add_model(rf_spec)

# -------------------------------------------------------
# 6) Hyperparameter grid for tuning
# -------------------------------------------------------
p <- length(setdiff(names(train), c("price", "property_id")))

# Common choices: sqrt(p), p/3, p/2
mtry_vals  <- unique(pmax(1, c(floor(sqrt(p)), floor(p/3), floor(p/2))))
min_n_vals <- c(5, 20, 50)

rf_grid <- tidyr::crossing(
  mtry  = mtry_vals,
  min_n = min_n_vals
)

print(rf_grid)

# -------------------------------------------------------
# 7) Spatial CV + tuning (parallelized)
# -------------------------------------------------------
cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)

control_rf <- control_grid(
  parallel_over = "resamples",  # Each fold runs in parallel
  verbose = TRUE
)

set.seed(2025)
rf_tuned <- tune_grid(
  wf_rf,
  resamples = block_folds,
  grid      = rf_grid,
  metrics   = metric_set(mae),
  control   = control_rf
)

# Stop parallel cluster
parallel::stopCluster(cl)
foreach::registerDoSEQ()

# Collect results
rf_metrics <- collect_metrics(rf_tuned)
print(rf_metrics)

# Save full tuning table
readr::write_csv(rf_metrics, "RF_spatialCV5_tuning_results.csv")

# -------------------------------------------------------
# 8) Select best hyperparameters (lowest MAE)
# -------------------------------------------------------
rf_best <- rf_metrics %>%
  filter(.metric == "mae") %>%
  arrange(mean) %>%
  slice(1)

print(rf_best)
readr::write_csv(rf_best, "RF_spatialCV5_best_parameters.csv")

best_params <- rf_best %>% select(mtry, min_n)

# -------------------------------------------------------
# 9) Fit final model with best params
# -------------------------------------------------------
wf_rf_final <- finalize_workflow(wf_rf, best_params)

set.seed(2025)
fit_rf <- fit(wf_rf_final, data = train)

# -------------------------------------------------------
# 10) Predict test set + Kaggle submission
# -------------------------------------------------------
pred_test <- predict(fit_rf, new_data = test)$.pred

# Round DOWN to nearest 1,000,000 COP
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = as.numeric(pred_test_round)
)

# File naming based on best hyperparameters
mtry_best  <- best_params$mtry
min_n_best <- best_params$min_n

file_name <- sprintf(
  "RF_spatialCV5_mtry%s_minN%s.csv",
  mtry_best,
  min_n_best
)

readr::write_csv(submission, file_name)

cat("\nSubmission saved as:", file_name, "\n")

# -------------------------------------------------------
# End of Script
# -------------------------------------------------------
