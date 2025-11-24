# =======================================================
# PS3 — Chapinero Prices
# Linear Regression + Spatial Cross-Validation (tidymodels)
# =======================================================

# -------------------------------------------------------
# 1) Load required libraries
# -------------------------------------------------------
# pacman installs packages automatically if missing
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,      # Data wrangling and manipulation
    tidymodels,     # Modeling framework: recipes, workflows, resampling
    sf,             # Simple features (spatial objects)
    spatialsample   # Spatial cross-validation: spatial_block_cv()
  )
})

# Avoid function conflicts between packages
tidymodels_prefer()

# -------------------------------------------------------
# 2) Load training and test datasets
# -------------------------------------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_train_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

test <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_test_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

# Basic integrity checks
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))  # Required for spatial CV

# Convert price to numeric (safety)
train$price <- as.numeric(train$price)

# -------------------------------------------------------
# 3) Build SPATIAL cross-validation folds (5 blocks)
# -------------------------------------------------------
# Spatial CV avoids overly optimistic model performance 
# when there is spatial autocorrelation in housing prices.

set.seed(2025)

# Convert training data into an sf spatial object, keeping all columns
train_sf <- sf::st_as_sf(
  train,
  coords = c("lon", "lat"),   # longitude/latitude coordinates
  crs    = 4326,
  remove = FALSE              # keep original columns in dataset
)

# Create 5 spatially separated folds
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# -------------------------------------------------------
# 4) Preprocessing pipeline: imputation + dummy variables
# -------------------------------------------------------
# Steps included:
#   • property_id is treated as an identifier (not a predictor)
#   • numerical variables → median imputation
#   • categorical variables → mode imputation
#   • convert categorical variables to dummy indicators

rec_lm <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%      # remove ID from predictors
  step_novel(all_nominal_predictors()) %>%           # handle unseen levels
  step_impute_median(all_numeric_predictors()) %>%   # numeric imputation
  step_impute_mode(all_nominal_predictors()) %>%     # categorical imputation
  step_dummy(all_nominal_predictors())               # one-hot encoding

# -------------------------------------------------------
# 5) Model specification: simple Linear Regression
# -------------------------------------------------------
lm_spec <- linear_reg() %>%
  set_engine("lm")   # Base R linear model

# Combine model + recipe into a workflow
wf_lm <- workflow() %>%
  add_model(lm_spec) %>%
  add_recipe(rec_lm)

# -------------------------------------------------------
# 6) Fit SPATIAL cross-validation and compute MAE
# -------------------------------------------------------
set.seed(2025)

res_lm <- fit_resamples(
  wf_lm,
  resamples = block_folds,               # 5 spatially separated folds
  metrics   = metric_set(mae),           # use MAE to evaluate accuracy
  control   = control_resamples(save_pred = TRUE)
)

# Extract and print average MAE across folds
lm_mae <- collect_metrics(res_lm) %>%
  filter(.metric == "mae")

print(lm_mae)
readr::write_csv(lm_mae, "LM_tm_spatialcv5_mae.csv")

# -------------------------------------------------------
# 7) Train final model using ALL training data
# -------------------------------------------------------
fit_lm <- fit(wf_lm, data = train)

# Optionally extract final set of predictors used after preprocessing
prep_rec <- prep(rec_lm, training = train)
baked_train <- bake(prep_rec, new_data = train)
vars_used <- setdiff(names(baked_train), "price")

readr::write_csv(tibble(var = vars_used), "LM_tm_vars_used.csv")

# -------------------------------------------------------
# 8) Predict on TEST dataset and create Kaggle submission
# -------------------------------------------------------
pred_test <- predict(fit_lm, new_data = test)$.pred

# Round predictions DOWN to the nearest 1,000,000 COP (competition rule)
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = as.numeric(pred_test_round)
)

print(head(submission))

# -------------------------------------------------------
# 9) Export submission file
# -------------------------------------------------------
model_label <- "LM_tm"       # Linear Model (tidymodels)
cv_label    <- "spatialcv5"  # Spatial 5-fold cross-validation

fname <- paste0(model_label, "_", cv_label, ".csv")
write.csv(submission, fname, row.names = FALSE)

cat("Submission file saved as:", fname, "\n")

# -------------------------------------------------------
# End of script
# -------------------------------------------------------
