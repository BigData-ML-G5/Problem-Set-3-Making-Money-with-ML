# =======================================================
# PS3 — Chapinero Prices
# SuperLearner (sl3) 
# =======================================================

# -------------------------------------------------------
# 0) Load packages
# -------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
library(pacman)

p_load(
  tidyverse,      # data wrangling
  sf,             # spatial objects (st_as_sf)
  spatialsample,  # spatial_block_cv for spatial folds
  glmnet,         # Elastic Net / LASSO engine
  ranger,         # Random Forest engine
  xgboost,        # Gradient Boosting engine
  nnls,           # non-negative least squares (metalearner)
  nnet,           # simple neural net
  earth,          # MARS
  rpart,          # CART
  future,         # parallel plans (we keep it sequential)
  sl3,            # SuperLearner framework
  origami         # folds infrastructure used by sl3
)

# To avoid huge global exports / memory issues
future::plan(sequential)
options(future.globals.maxSize = 8 * 1024^3)

# -------------------------------------------------------
# 1) Load final train & test data (already preprocessed)
# -------------------------------------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

# Drop property_type if present (not used as predictor)
if ("property_type" %in% names(train)) {
  train <- train %>% dplyr::select(-property_type)
}
if ("property_type" %in% names(test)) {
  test  <- test  %>% dplyr::select(-property_type)
}

# IDs (kept only for merging predictions later)
train_id <- train$property_id
test_id  <- test$property_id

# Ensure numeric outcome
train$price <- as.numeric(train$price)

# Some Kaggle test files may carry a dummy price; we drop it if present
if ("price" %in% names(test)) {
  test <- test %>% dplyr::select(-price)
}

# Basic sanity checks
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon", "lat") %in% names(train)))  # needed for spatial CV

# -------------------------------------------------------
# 2) Spatial CV folds (spatial_block_cv + origami)
# -------------------------------------------------------
set.seed(2025)

# Convert to sf object using lon/lat, keeping all original columns
train_sf <- sf::st_as_sf(
  train,
  coords = c("lon", "lat"),
  crs    = 4326,
  remove = FALSE
)

# Create 5 spatial blocks (each fold is a spatial cluster)
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

num_obs <- nrow(train)
vec_obs <- 1:num_obs
fold_id_df <- data.frame(ID = integer(), num_fold = integer())

# For each block, define which observations are in the *assessment* set
for (k in seq_along(block_folds$splits)) {
  temp_id <- setdiff(vec_obs, block_folds$splits[[k]][["in_id"]])
  temp_db <- data.frame(ID = temp_id, num_fold = k)
  fold_id_df <- dplyr::bind_rows(fold_id_df, temp_db)
}

fold_id_df <- fold_id_df |> dplyr::arrange(ID)
fold_id <- fold_id_df$num_fold

# Build origami folds object (used internally by sl3)
folds <- origami::make_folds(cluster_ids = fold_id)

# -------------------------------------------------------
# 3) Define X and y WITHOUT heavy preprocessing
# -------------------------------------------------------
# We rely on train_unified_final already being cleaned and imputed.
# We only:
#   - Exclude 'price' and 'property_id' from X
#   - Align factor levels between train and test

# Candidate covariates = all columns except outcome and ID
x_cols <- setdiff(names(train), c("price", "property_id"))

# Check that test has the same covariates
stopifnot(all(x_cols %in% names(test)))

# Extract raw covariate matrices
train_x_raw <- train[, x_cols, drop = FALSE]
test_x_raw  <- test[,  x_cols, drop = FALSE]

# Bind train + test to ensure consistent factor levels
full_x <- dplyr::bind_rows(
  train_x_raw %>% dplyr::mutate(.is_train = TRUE),
  test_x_raw  %>% dplyr::mutate(.is_train = FALSE)
)

# Convert character columns to factors (minimum necessary)
char_cols <- names(which(sapply(full_x, is.character)))
full_x[char_cols] <- lapply(full_x[char_cols], factor)

# Split back into train and test design matrices
X_train <- full_x %>%
  dplyr::filter(.is_train) %>%
  dplyr::select(-.is_train)

X_test  <- full_x %>%
  dplyr::filter(!.is_train) %>%
  dplyr::select(-.is_train)

Y_train <- train$price

cat("X_train dimensions:", dim(X_train), "\n")
cat("X_test  dimensions:", dim(X_test),  "\n")
cat("Any NA in X_train?:", anyNA(X_train), "\n")
cat("Any NA in X_test ?: ", anyNA(X_test), "\n")

# -------------------------------------------------------
# 4) Build sl3 Task using raw data + spatial folds
# -------------------------------------------------------
train_for_task <- data.frame(
  price = Y_train,
  X_train
)

task_train <- sl3::sl3_Task$new(
  data       = train_for_task,
  covariates = x_cols,
  outcome    = "price",
  folds      = folds    # spatial folds from origami
)

cat("Number of covariates in SL3 task:", length(x_cols), "\n")

# -------------------------------------------------------
# 5) Define base learners (with tuned hyperparameters)
# -------------------------------------------------------

# 5.1 GLMNET (LASSO)
# Let glmnet internally choose lambda; alpha = 1 enforces LASSO penalty.
lrn_glmnet <- sl3::Lrnr_glmnet$new(
  alpha       = 1,
  standardize = TRUE    # standardization handled inside glmnet
)

# 5.2 Random Forest (ranger) using tuned hyperparameters
# From your RF tuning: num.trees = 800, min.node.size = 5, mtry = 65
lrn_ranger <- sl3::Lrnr_ranger$new(
  num.trees     = 800,
  min.node.size = 5,
  mtry          = min(65, length(x_cols))  # in case p < 65
)

# 5.3 XGBoost using your tuned hyperparameters
lrn_xgb <- sl3::Lrnr_xgboost$new(
  nrounds          = 500,
  max_depth        = 10,
  eta              = 0.03,
  subsample        = 0.9,
  colsample_bytree = 0.8,
  min_child_weight = 20,
  objective        = "reg:squarederror"
)

# 5.4 MARS (earth)
lrn_earth <- sl3::Lrnr_earth$new()

# 5.5 CART (rpart)
lrn_rpart <- sl3::Lrnr_rpart$new()

# 5.6 Shallow Neural Net (nnet)
lrn_nnet <- sl3::Lrnr_nnet$new(
  size    = 5,
  linout  = TRUE,
  maxit   = 300,
  trace   = FALSE,
  MaxNWts = 5000
)

# 5.7 Simple mean model (benchmark)
lrn_mean <- sl3::Lrnr_mean$new()

# Stack of base learners
learners <- sl3::Stack$new(
  lrn_glmnet,
  lrn_ranger,
  lrn_xgb,
  lrn_earth,
  lrn_rpart,
  lrn_nnet,
  lrn_mean
)

# Metalearner: Non-negative least squares (NNLS)
# Ensures convex combination of base learners.
metalearner <- sl3::Lrnr_nnls$new()

# Final SuperLearner
sl <- sl3::Lrnr_sl$new(
  learners    = learners,
  metalearner = metalearner
)

# -------------------------------------------------------
# 6) Train SuperLearner
# -------------------------------------------------------
set.seed(2025)
sl_fit <- sl$train(task = task_train)

cat("\nSuperLearner training completed.\n")

# -------------------------------------------------------
# 7) SuperLearner ensemble weights
# -------------------------------------------------------
# Coefficients are the weights assigned to each base learner
sl_coefs <- sl_fit$coef

cat("\nSuperLearner weights (non-negative, sum to 1):\n")
print(sl_coefs)

# Save weights to CSV for documentation
weights_df <- tibble(
  learner = names(sl_coefs),
  weight  = as.numeric(sl_coefs)
)

readr::write_csv(weights_df, "SuperLearner_S3_sl3_spatial_weights.csv")

# -------------------------------------------------------
# 8) In-sample MAE (approximate)
# -------------------------------------------------------
pred_train_sl <- sl_fit$predict(task = task_train)
mae_train_sl  <- mean(abs(pred_train_sl - Y_train), na.rm = TRUE)

cat("\nApproximate in-sample MAE (SL3):", mae_train_sl, "\n")

# -------------------------------------------------------
# 9) Test predictions + Kaggle submission
# -------------------------------------------------------
task_test <- sl3::sl3_Task$new(
  data       = X_test,
  covariates = x_cols
)

pred_test_sl <- sl_fit$predict(task = task_test)
pred_test_sl <- as.numeric(pred_test_sl)

# Round down to nearest million (Kaggle convention for this PS)
pred_test_sl_round <- floor(pred_test_sl / 1e6) * 1e6

submission_sl <- tibble(
  property_id = test_id,
  price       = pred_test_sl_round
)

print(head(submission_sl))

model_label <- "SuperLearner_S3_sl3_raw_tuned"
cv_label    <- "spatialcv5"

fname_sl <- paste0(model_label, "_", cv_label, ".csv")

write.csv(submission_sl, fname_sl, row.names = FALSE)
cat("\nSubmission file saved as:", fname_sl, "\n")

# -------------------------------------------------------
# 10) Save list of variables used by SL3
# -------------------------------------------------------
vars_used_sl <- x_cols

write.csv(
  data.frame(var = vars_used_sl),
  "SuperLearner_S3_sl3_raw_tuned_vars_used.csv",
  row.names = FALSE
)

cat("\n========== SUPERLEARNER S3 (sl3, NO MANUAL PREPRO, TUNED) FINISHED ==========\n")
