# ============================================
# 1. Librerías
# ============================================
library(dplyr)
library(xgboost)
library(caret)

# ============================================
# 2. Selección de features numéricos
# ============================================
num_vars <- train_final %>%
  select(-property_id, -price) %>%  
  select_if(is.numeric) %>% 
  names()

dtrain <- xgb.DMatrix(
  data = as.matrix(train_final[, num_vars]),
  label = train_final$price
)

dtest <- xgb.DMatrix(
  data = as.matrix(test_final[, num_vars])
)

# ============================================
# 3. Hiperparámetros
# ============================================
params <- list(
  objective = "reg:squarederror",
  booster = "gbtree",
  eta = 0.06,
  max_depth = 7,
  subsample = 0.85,
  colsample_bytree = 0.75,
  min_child_weight = 5,
  lambda = 2,
  eval_metric = "rmse"
)

# ============================================
# 4. Validación cruzada espacial (K-means)
# ============================================
set.seed(123)

colnames(train_final)
# Clustering espacial con long/lat
kmeans_space <- kmeans(train_final[, c("lon", "lat")], centers = 5)
train_final$spatial_fold <- kmeans_space$cluster

# Generar folds espaciales
folds <- groupKFold(train_final$spatial_fold)

rmse_list <- c()

for (i in 1:length(folds)) {
  
  train_idx <- folds[[i]]
  test_idx  <- setdiff(seq_len(nrow(train_final)), train_idx)
  
  dtrain_cv <- xgb.DMatrix(
    data = as.matrix(train_final[train_idx, num_vars]),
    label = train_final$price[train_idx]
  )
  
  dtest_cv <- xgb.DMatrix(
    data = as.matrix(train_final[test_idx, num_vars]),
    label = train_final$price[test_idx]
  )
  
  # Entrenamiento fold
  model_cv <- xgb.train(
    params = params,
    data = dtrain_cv,
    nrounds = 650,
    verbose = 0
  )
  
  preds <- predict(model_cv, dtest_cv)
  
  rmse_list[i] <- sqrt(mean((preds - train_final$price[test_idx])^2))
  
  cat("Fold", i, "- RMSE:", round(rmse_list[i], 2), "\n")
}

cat("\nRMSE espacial promedio:", round(mean(rmse_list), 2), "\n\n")

# ============================================
# 5. Entrenamiento final del modelo
# ============================================
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 650,
  watchlist = list(train = dtrain),
  print_every_n = 50
)

# ============================================
# 6. Predicciones para Kaggle
# ============================================
test_final$pred_price <- predict(xgb_model, dtest)

kaggle_submission <- data.frame(
  property_id = test_final$property_id,
  price = test_final$pred_price
)

write.csv(kaggle_submission,
          "submission_xgb_spatialCV.csv",
          row.names = FALSE)

cat("Archivo generado: submission_xgb_spatialCV.csv\n")
