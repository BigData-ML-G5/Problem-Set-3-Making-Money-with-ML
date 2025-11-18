library(xgboost)
library(dplyr)

# 1. Selección de features numéricos
num_vars <- train_final %>%
  select(-property_id, -price) %>%     # cambia price por el nombre exacto de tu target
  select_if(is.numeric) %>% 
  names()

dtrain <- xgb.DMatrix(
  data = as.matrix(train_final[, num_vars]),
  label = train_final$price
)

dtest <- xgb.DMatrix(
  data = as.matrix(test_final[, num_vars])
)

# 2. Hiperparámetros recomendados
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

# 3. Entrenamiento con validación interna (no Kaggle)
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 650,
  watchlist = list(train = dtrain),
  print_every_n = 50
)

# 4. Predicción para Kaggle (solo Chapinero)
test_final$pred_price <- predict(xgb_model, dtest)

# 5. Crear archivo Kaggle
kaggle_submission <- data.frame(
  property_id = test_final$property_id,
  price = test_final$pred_price
)

write.csv(kaggle_submission,
          "submission_xgb.csv",
          row.names = FALSE)

summary(train_final$price)
