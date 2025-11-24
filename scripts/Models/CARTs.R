# =======================================================
# PS3 — Chapinero Prices
# CART (Regression Tree) with Spatial Cross-Validation
# =======================================================

# -------------------------------------------------------
# 1) Load libraries
# -------------------------------------------------------
# pacman installs packages if they are missing and then loads them
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,   # data manipulation
    caret,       # modeling framework (tuning, CV)
    rpart,       # CART implementation
    sf,          # spatial objects (simple features)
    spatialsample, # spatial_block_cv for spatial CV
    rsample,     # analysis()/assessment() helpers
    forcats      # factor handling (fct_explicit_na, fct_other)
  )
})

# -------------------------------------------------------
# 2) Load data (FINAL unified datasets)
# -------------------------------------------------------
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
stopifnot(all(c("lon","lat") %in% names(train)))  # required for spatial CV

# Ensure target is numeric
train$price <- as.numeric(train$price)

# -------------------------------------------------------
# 3) Column selection & simple imputation
# -------------------------------------------------------
# We drop ID and outcome from the predictor set
cols_drop    <- c("property_id", "price")
x_cols_train <- setdiff(names(train), cols_drop)
x_cols_test  <- setdiff(names(test),  "property_id")

# Keep only predictors that appear in BOTH train and test
common_cols  <- intersect(x_cols_train, x_cols_test)
if (length(common_cols) == 0) {
  stop("No common predictors between train and test. Check column names.")
}
x_cols <- common_cols

# Modeling data:
#   - train_df: price + predictors
#   - test_df : property_id + predictors (for submission)
train_df <- train %>% dplyr::select(price, all_of(x_cols))
test_df  <- test  %>% dplyr::select(property_id, all_of(x_cols))

# Helper: mode for categorical variables
impute_mode <- function(v) {
  tb <- table(v, useNA = "no")
  names(tb)[which.max(tb)]
}

# Simple imputation:
#   - Numeric: median
#   - Categorical: mode, align levels between train and test
for (nm in names(train_df)) {
  if (nm == "price") next  # do not touch target
  
  if (is.numeric(train_df[[nm]])) {
    # Numeric: median imputation
    med <- median(train_df[[nm]], na.rm = TRUE)
    train_df[[nm]][is.na(train_df[[nm]])] <- med
    if (nm %in% names(test_df)) {
      test_df[[nm]][is.na(test_df[[nm]])] <- med
    }
  } else {
    # Categorical: factor + mode + aligned levels
    train_df[[nm]] <- as.factor(train_df[[nm]])
    mode_val <- impute_mode(train_df[[nm]])
    train_df[[nm]][is.na(train_df[[nm]])] <- mode_val
    
    if (nm %in% names(test_df)) {
      test_df[[nm]] <- as.factor(test_df[[nm]])
      # Make explicit NAs and collapse unknown levels into "mode_val"
      test_df[[nm]] <- forcats::fct_explicit_na(test_df[[nm]], na_level = mode_val)
      test_df[[nm]] <- forcats::fct_other(
        test_df[[nm]],
        keep        = levels(train_df[[nm]]),
        other_level = mode_val
      )
      # Force same levels as in train
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
      test_df[[nm]][is.na(test_df[[nm]])] <- mode_val
    }
  }
}

# -------------------------------------------------------
# 4) Spatial CV folds (5-fold block CV)
# -------------------------------------------------------
# We create spatial folds using lon/lat coordinates to avoid
# overly optimistic performance in the presence of spatial dependence.

set.seed(2025)

# Convert to sf using lon/lat; train_df MUST contain lon and lat
train_sf <- sf::st_as_sf(
  train_df,
  coords = c("lon", "lat"),
  crs    = 4326
)

# Create 5 spatially separated folds
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# Convert rsample splits into index lists for caret
index_list <- lapply(block_folds$splits, function(s) {
  which(rownames(train_df) %in% rownames(rsample::analysis(s)))
})

indexOut_list <- lapply(block_folds$splits, function(s) {
  which(rownames(train_df) %in% rownames(rsample::assessment(s)))
})

# -------------------------------------------------------
# 5) MAE summary function + caret control object
# -------------------------------------------------------
# Custom summary function to report MAE during training
maeSummary <- function(data, lev = NULL, model = NULL) {
  c(MAE = caret::MAE(pred = data$pred, obs = data$obs))
}

# TrainControl object specifying:
#   - cross-validation with our custom spatial indices
#   - MAE as evaluation metric
ctrl_spatial <- caret::trainControl(
  method          = "cv",
  number          = length(index_list),  # should be 5
  summaryFunction = maeSummary,
  index           = index_list,
  indexOut        = indexOut_list,
  verboseIter     = FALSE
)

# -------------------------------------------------------
# 6) CART (rpart) tuning over complexity parameter cp
# -------------------------------------------------------
# cp controls how aggressively the tree is pruned
set.seed(2025)
cart_cp <- caret::train(
  price ~ .,
  data      = train_df,
  method    = "rpart",  # complexity-parameter-based CART
  metric    = "MAE",
  trControl = ctrl_spatial,
  tuneGrid  = expand.grid(cp = seq(0.0005, 0.01, length.out = 20))
)

# -------------------------------------------------------
# 7) CART (rpart2) tuning over maximum depth
# -------------------------------------------------------
# rpart2 tunes the maximum depth of the tree directly
set.seed(2025)
cart_depth <- caret::train(
  price ~ .,
  data      = train_df,
  method    = "rpart2",       # depth-based CART
  metric    = "MAE",
  trControl = ctrl_spatial,
  tuneGrid  = expand.grid(maxdepth = 1:15)
)

# -------------------------------------------------------
# 8) Compare both CART models and select the best one
# -------------------------------------------------------
mae_cp <- cart_cp$results$MAE[
  cart_cp$results$cp == cart_cp$bestTune$cp
]

mae_depth <- cart_depth$results$MAE[
  cart_depth$results$maxdepth == cart_depth$bestTune$maxdepth
]

if (mae_cp <= mae_depth) {
  best_model <- cart_cp
  best_type  <- "cp"
} else {
  best_model <- cart_depth
  best_type  <- "depth"
}

cat("\nBest CART model type:", best_type, "\n")
cat("MAE (cp-based):   ", mae_cp, "\n")
cat("MAE (depth-based):", mae_depth, "\n")

# -------------------------------------------------------
# 9) Predict on TEST set + create Kaggle submission
# -------------------------------------------------------
# Use ONLY predictors (x_cols) for testing
pred_test <- predict(best_model, newdata = test_df[, x_cols])

# Round predictions DOWN to the nearest million COP
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test_df$property_id,
  price       = as.numeric(pred_test_round)
)

# Name the output file based on which model was selected
if (best_type == "cp") {
  cp_val  <- best_model$bestTune$cp
  cp_str  <- gsub("\\.", "_", signif(cp_val, 5))
  fname   <- paste0("CART_cp_", cp_str, "_spatialcv5.csv")
} else {
  depth_val <- best_model$bestTune$maxdepth
  fname     <- paste0("CART_depth_", depth_val, "_spatialcv5.csv")
}

write.csv(submission, fname, row.names = FALSE)
cat("\nSubmission file saved as:", fname, "\n")

# -------------------------------------------------------
# End of script
# -------------------------------------------------------

