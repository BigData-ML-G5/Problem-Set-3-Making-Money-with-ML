# =======================================================
# PS3 — Chapinero Prices
# CART (Regression Tree) + Spatial CV (FAST VERSION)
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    caret,
    rpart,
    sf,
    spatialsample,
    rsample
  )
})

# ------------------------
# 2) Load data
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_train_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_test_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

# Checks
stopifnot("property_id" %in% names(train))
stopifnot("price"       %in% names(train))
stopifnot(all(c("lon","lat") %in% names(train)))

train$price <- as.numeric(train$price)

# ------------------------
# 3) Column selection & simple imputation
# ------------------------
cols_drop    <- c("property_id", "price")
x_cols_train <- setdiff(names(train), cols_drop)
x_cols_test  <- setdiff(names(test),  "property_id")

common_cols  <- intersect(x_cols_train, x_cols_test)
x_cols       <- common_cols

train_df <- train %>% select(price, all_of(x_cols))
test_df  <- test  %>% select(property_id, all_of(x_cols))

# Mode function
impute_mode <- function(v) names(which.max(table(v)))

# Simple imputations
for (nm in names(train_df)) {
  if (nm == "price") next
  
  if (is.numeric(train_df[[nm]])) {
    med <- median(train_df[[nm]], na.rm = TRUE)
    train_df[[nm]][is.na(train_df[[nm]])] <- med
    if (nm %in% names(test_df)) test_df[[nm]][is.na(test_df[[nm]])] <- med
  } else {
    train_df[[nm]] <- as.factor(train_df[[nm]])
    mode_val <- impute_mode(train_df[[nm]])
    train_df[[nm]][is.na(train_df[[nm]])] <- mode_val
    
    if (nm %in% names(test_df)) {
      test_df[[nm]] <- as.factor(test_df[[nm]])
      test_df[[nm]] <- forcats::fct_explicit_na(test_df[[nm]], na_level = mode_val)
      test_df[[nm]] <- forcats::fct_other(
        test_df[[nm]],
        keep        = levels(train_df[[nm]]),
        other_level = mode_val
      )
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
    }
  }
}

# ------------------------
# 4) Spatial CV folds
# ------------------------
set.seed(2025)

train_sf <- sf::st_as_sf(
  train_df,
  coords = c("lon", "lat"),
  crs    = 4326
)

block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

index_list <- lapply(block_folds$splits, function(s)
  which(rownames(train_df) %in% rownames(rsample::analysis(s))))

indexOut_list <- lapply(block_folds$splits, function(s)
  which(rownames(train_df) %in% rownames(rsample::assessment(s))))

# ------------------------
# 5) MAE summary + caret control
# ------------------------
maeSummary <- function(data, lev = NULL, model = NULL) {
  c(MAE = caret::MAE(data$pred, data$obs))
}

ctrl_spatial <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = maeSummary,
  index           = index_list,
  indexOut        = indexOut_list,
  verboseIter     = FALSE
)

# ------------------------
# 6) CART_rpart (tuning cp)
# ------------------------
set.seed(2025)
cart_cp <- train(
  price ~ .,
  data      = train_df,
  method    = "rpart",
  metric    = "MAE",
  trControl = ctrl_spatial,
  tuneGrid  = expand.grid(cp = seq(0.0005, 0.01, length.out = 20))
)

# ------------------------
# 7) CART_rpart2 (tuning maxdepth)
# ------------------------
set.seed(2025)
cart_depth <- train(
  price ~ .,
  data      = train_df,
  method    = "rpart2",
  metric    = "MAE",
  trControl = ctrl_spatial,
  tuneGrid  = expand.grid(maxdepth = 1:15)
)

# ------------------------
# 8) Compare & pick best model
# ------------------------
mae_cp    <- cart_cp$results$MAE[cart_cp$results$cp == cart_cp$bestTune$cp]
mae_depth <- cart_depth$results$MAE[cart_depth$results$maxdepth == cart_depth$bestTune$maxdepth]

if (mae_cp <= mae_depth) {
  best_model <- cart_cp
  best_type  <- "cp"
} else {
  best_model <- cart_depth
  best_type  <- "depth"
}

# ------------------------
# 9) Predict TEST + save submission
# ------------------------
pred_test <- predict(best_model, newdata = test_df[, x_cols])

pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test_df$property_id,
  price       = pred_test_round
)

if (best_type == "cp") {
  cp_val  <- best_model$bestTune$cp
  cp_str  <- gsub("\\.", "_", signif(cp_val, 5))
  fname   <- paste0("CART_cp_", cp_str, "_spatialcv5.csv")
} else {
  depth_val <- best_model$bestTune$maxdepth
  fname     <- paste0("CART_depth_", depth_val, "_spatialcv5.csv")
}

write.csv(submission, fname, row.names = FALSE)

cat("Submission guardada como:", fname, "\n")
