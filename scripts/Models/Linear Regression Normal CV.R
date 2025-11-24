# =======================================================
# PS3 — Chapinero Prices
# Linear Regression + Standard 5-fold CV 
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels   # recipes, parsnip, workflows, rsample, yardstick
  )
})

tidymodels_prefer()

# ------------------------
# 2) Load data (FINAL unified data)
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

# Basic checks
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))

# Make sure price is numeric
train$price <- as.numeric(train$price)

# ------------------------
# 3) Recipe: imputation + dummies
# ------------------------
rec_lm <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%     # do not use as predictor
  step_novel(all_nominal_predictors()) %>%          # handle unseen levels
  step_impute_median(all_numeric_predictors()) %>%  # numeric imputation
  step_impute_mode(all_nominal_predictors()) %>%    # categorical imputation
  step_dummy(all_nominal_predictors())              # one-hot encoding

# ------------------------
# 4) Model specification: simple Linear Regression
# ------------------------
lm_spec <- linear_reg() %>%
  set_engine("lm")

wf_lm <- workflow() %>%
  add_model(lm_spec) %>%
  add_recipe(rec_lm)

# ------------------------
# 5) STANDARD (non-spatial) 5-fold cross-validation
#     (just to report MAE)
# ------------------------
set.seed(2025)
folds_standard <- vfold_cv(train, v = 5)

res_lm <- fit_resamples(
  wf_lm,
  resamples = folds_standard,
  metrics   = metric_set(mae),
  control   = control_resamples(save_pred = TRUE)
)

lm_mae <- collect_metrics(res_lm) %>%
  filter(.metric == "mae")

print(lm_mae)
readr::write_csv(lm_mae, "LM_unified_standardcv5_mae.csv")

# ------------------------
# 6) Fit final model on FULL training data
# ------------------------
fit_lm <- fit(wf_lm, data = train)

# ------------------------
# 7) Predict on TEST and build Kaggle submission
# ------------------------
pred_test <- predict(fit_lm, new_data = test)$.pred

# Round down to the nearest million
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = as.numeric(pred_test_round)
)

print(head(submission))

# ------------------------
# 8) Save submission
# ------------------------
model_label <- "LM_unified"
cv_label    <- "standardcv5"

fname <- paste0(model_label, "_", cv_label, ".csv")
write.csv(submission, fname, row.names = FALSE)
cat("Submission saved as: ", fname, "\n")

# ------------------------
# End of script
# ------------------------
