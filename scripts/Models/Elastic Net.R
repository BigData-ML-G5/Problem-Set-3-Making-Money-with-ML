# =======================================================
# PS3 — Chapinero Prices
# Elastic Net 
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse, dplyr, tidyr, readr, ggplot2, forcats,
    caret,         # high-level modeling framework + CV
    glmnet,        # Elastic Net engine
    sf,            # st_as_sf for spatial objects
    spatialsample, # spatial_block_cv for spatial CV
    rsample        # analysis()/assessment() helpers for rsample objects
  )
})

# ------------------------
# 2) Load data (NEW unified paths)
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

# Drop property_type if present (not used as predictor)
train <- train %>% select(-property_type)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test <- test %>% select(-property_type)

# Basic checks: IDs, outcome, and spatial coordinates
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))  # needed for spatial CV

# ------------------------
# 3) Target and predictor selection
# ------------------------
# Ensure price is numeric for glmnet
train$price <- as.numeric(train$price)

# Columns we do NOT want as predictors
cols_drop    <- c("property_id", "price")

x_cols_train <- setdiff(names(train), cols_drop)
x_cols_test  <- setdiff(names(test),  "property_id")

# Keep only predictors that appear in both train and test
common_cols  <- intersect(x_cols_train, x_cols_test)
if (length(common_cols) == 0) {
  stop("No common predictors between train and test. Check column names.")
}

x_cols <- common_cols

# Final modeling frames:
# - train_df: outcome + predictors
# - test_df : ID + predictors (for Kaggle submission)
train_df <- train %>%
  dplyr::select(all_of(c("price", x_cols)))

test_df  <- test %>%
  dplyr::select(all_of(c("property_id", x_cols)))

# ------------------------
# 4) Simple NA imputation and type handling
#    (numeric: median; factors: mode + aligned levels)
# ------------------------

# Helper function to impute the mode for categorical variables
impute_mode <- function(v) {
  tb <- table(v, useNA = "no")
  names(tb)[which.max(tb)]
}

for (nm in names(train_df)) {
  # Skip the outcome
  if (nm == "price") next
  
  # Numeric predictors: median imputation
  if (is.numeric(train_df[[nm]])) {
    med <- median(train_df[[nm]], na.rm = TRUE)
    train_df[[nm]][is.na(train_df[[nm]])] <- med
    
    if (nm %in% names(test_df)) {
      test_df[[nm]][is.na(test_df[[nm]])] <- med
    }
    
  } else {
    # Categorical / character predictors: convert to factor + mode imputation
    train_df[[nm]] <- as.factor(train_df[[nm]])
    mode_val <- impute_mode(train_df[[nm]])
    train_df[[nm]][is.na(train_df[[nm]])] <- mode_val
    
    if (nm %in% names(test_df)) {
      test_df[[nm]] <- as.factor(test_df[[nm]])
      
      # Make NA explicit, then collapse "new" levels into an "other" = mode_val
      test_df[[nm]] <- forcats::fct_explicit_na(test_df[[nm]], na_level = mode_val)
      test_df[[nm]] <- forcats::fct_other(
        test_df[[nm]],
        keep        = levels(train_df[[nm]]),
        other_level = mode_val
      )
      
      # Align factor levels between train and test
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
      test_df[[nm]][is.na(test_df[[nm]])] <- mode_val
    }
  }
}

# ------------------------
# 5) Spatial Cross-Validation (5-fold) + MAE
# ------------------------
set.seed(2025)

# Convert train_df to an sf object using lon/lat.
# Note: lon and lat must still be present in train_df as predictors.
train_sf <- sf::st_as_sf(
  train_df,
  coords = c("lon", "lat"),
  crs    = 4326
)

# Create spatial blocks (folds) based on geographic structure
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# Convert rsample object into index lists for caret::train()
index_list <- lapply(block_folds$splits, function(s) {
  which(rownames(train_df) %in% rownames(rsample::analysis(s)))
})

indexOut_list <- lapply(block_folds$splits, function(s) {
  which(rownames(train_df) %in% rownames(rsample::assessment(s)))
})

# Custom MAE summary function for caret
maeSummary <- function(data, lev = NULL, model = NULL) {
  c(MAE = caret::MAE(pred = data$pred, obs = data$obs))
}

# Custom trainControl object for SPATIAL CV
ctrl_spatial_en <- trainControl(
  method          = "cv",          # caret still thinks "cv", but indices are spatial
  number          = length(index_list),  # v = 5 folds
  summaryFunction = maeSummary,
  index           = index_list,    # training indices per fold (spatial)
  indexOut        = indexOut_list, # test/assessment indices per fold (spatial)
  verboseIter     = TRUE
)

# ------------------------
# 6) Hyperparameter grid for Elastic Net (alpha, lambda)
# ------------------------
# alpha: mixing parameter between Ridge (0) and Lasso (1)
# lambda: regularization strength
tune_grid_en <- expand.grid(
  alpha  = seq(0.2, 1.0, by = 0.2),            # 0.2, 0.4, ..., 1.0
  lambda = seq(0.0005, 0.01, length.out = 10)  # can adjust if needed
)

# ------------------------
# 7) Train Elastic Net (glmnet) with SPATIAL CV
# ------------------------
set.seed(2025)
model_en <- caret::train(
  price ~ .,
  data       = train_df,
  method     = "glmnet",
  trControl  = ctrl_spatial_en,
  tuneGrid   = tune_grid_en,
  metric     = "MAE",                    # optimize Mean Absolute Error
  preProcess = c("center", "scale"),     # standardize predictors for glmnet
  family     = "gaussian"                # regression
)

# Print the full model summary and best hyperparameters
print(model_en)
print(model_en$bestTune)

# Save full CV results (MAE by alpha-lambda combination)
cv_results_en <- model_en$results
readr::write_csv(cv_results_en, "EN_caret_spatialcv5_tuning_mae.csv")

# Extract row corresponding to the best model
best_row_en <- cv_results_en %>%
  dplyr::filter(
    alpha  == model_en$bestTune$alpha,
    lambda == model_en$bestTune$lambda
  )
print(best_row_en)
readr::write_csv(best_row_en, "EN_caret_spatialcv5_best_mae.csv")

# ------------------------
# 8) Predictions on TEST set + Kaggle submission
# ------------------------
pred_price_en <- predict(
  model_en,
  newdata = test_df %>% dplyr::select(all_of(x_cols))
)

if (any(is.na(pred_price_en))) {
  warning("There are NAs in the predictions; please check imputation and factor levels.")
}

# Round down to the nearest million (COP, same convention as XGBoost script)
pred_price_en_round <- floor(pred_price_en / 1e6) * 1e6

submission_en <- tibble(
  property_id = test_df$property_id,
  price       = as.numeric(pred_price_en_round)
)

print(head(submission_en))

# ------------------------
# 9) Name output file according to best hyperparameters
# ------------------------
lambda_val <- model_en$bestTune$lambda
alpha_val  <- model_en$bestTune$alpha

lambda_str <- gsub("\\.", "_", format(round(lambda_val, 6), scientific = FALSE))
alpha_str  <- gsub("\\.", "_", format(round(alpha_val, 3), scientific = FALSE))

model_label <- "EN_caret"
cv_label    <- "spatialcv5"

fname_en <- paste0(
  model_label, "_", cv_label,
  "_lambda_", lambda_str,
  "_alpha_", alpha_str,
  ".csv"
)

write.csv(submission_en, fname_en, row.names = FALSE)
cat("Submission file saved as: ", fname_en, "\n")

# ------------------------
# 10) Save list of predictors used
# ------------------------
vars_used_en <- x_cols
write.csv(
  data.frame(var = vars_used_en),
  "EN_caret_vars_used.csv",
  row.names = FALSE
)

cat("\n========== Elastic Net (caret + glmnet + spatial CV) FINISHED ==========\n")

# ------------------------
# End of script
# ------------------------
