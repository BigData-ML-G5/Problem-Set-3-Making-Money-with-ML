
# ============================================================
# 0. LOAD PACKAGES
# ============================================================
require(pacman)

p_load(dplyr, recipes, SuperLearner, sf, sp, earth, ranger, glmnet, xgboost)


# ============================================================
# 1. LOAD DATA
# ============================================================
train <- read.csv(
  # TODO
  "//CODD.sis.virtual.uniandes.edu.co/Estudiantes/Profiles/k.gonzalezj/Documents/ps3/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

train <- train %>% select(-property_type)

test  <- read.csv(
  # TODO
  "//CODD.sis.virtual.uniandes.edu.co/Estudiantes/Profiles/k.gonzalezj/Documents/ps3/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test <- test %>% select(-property_type, -price)


# ============================================================
# 2. SAVE IDS BEFORE PROCESSING
# ============================================================
train_id <- train$property_id
test_id  <- test$property_id


# ============================================================
# 3. REMOVE ID FROM TRAINING PREDICTORS
# ============================================================
train <- train %>% select(-property_id)
test  <- test  %>% select(-property_id)


# ============================================================
# 4. RECIPE FOR PREPROCESSING
# ============================================================
rec <- recipe(price ~ ., data = train) %>%
  
  step_nzv(all_predictors()) %>%
  
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())


# ============================================================
# 5. PREP & BAKE
# ============================================================
rec_prep <- prep(rec, training = train, retain = TRUE)

train_baked <- bake(rec_prep, new_data = train)
test_baked  <- bake(rec_prep, new_data = test)

print(anyNA(train_baked))
print(anyNA(test_baked))
print(ncol(train_baked))


# ============================================================
# 6. SPATIAL CV FOLDS
# ============================================================
coords <- train %>% select(lat, lon)
sp_points <- SpatialPoints(coords)

K <- 5
set.seed(123)
clusters <- kmeans(coords, centers = K)$cluster

validRows_clean <- lapply(1:K, function(k) which(clusters == k))
print(sapply(validRows_clean, length))

cvControl <- list(
  V = length(validRows_clean),
  validRows = validRows_clean
)


# ============================================================
# 7. SUPERLEARNER
# ============================================================
Y <- train_baked$price
X <- train_baked %>% select(-price)

sl_lib <- c(
  "SL.xgboost",  # XGBoost
  "SL.glmnet",
  "SL.ranger",
  "SL.earth",
  "SL.rpart",
  "SL.mean"
)

sl_fit <- SuperLearner(
  Y = Y,
  X = X,
  SL.library = sl_lib,
  cvControl = cvControl,
  family = gaussian()
)


# ============================================================
# 8. PREDICTIONS ON TEST SET
# ============================================================
pred <- predict(sl_fit, newdata = test_baked)$pred


# ============================================================
# 9. FINAL SUBMISSION FILE (ID + PRICE)
# ============================================================
output <- data.frame(
  property_id = test_id,
  price = pred
)

write.csv(output, "prediccion_final.csv", row.names = FALSE)

cat("\n========== SUPERLEARNER FINALIZADO ==========\n")