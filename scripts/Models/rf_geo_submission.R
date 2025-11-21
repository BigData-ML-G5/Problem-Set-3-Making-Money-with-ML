library(ranger)
library(dplyr)

# Seleccionar features numéricas (sin ID ni target)
predictors <- train_final %>%
  select_if(is.numeric) %>%
  select(-price, -property_id) %>%
  names()

# Modelo Random Forest muy simple
rf_model <- ranger(
  formula = price ~ .,
  data = train_final[, c("price", predictors)],
  num.trees = 600,
  mtry = floor(sqrt(length(predictors))),  # regla clásica
  min.node.size = 5,
  importance = "impurity",
  seed = 2025
)

# Predicción en test
test_final$pred_rf <- predict(rf_model, test_final)$predictions

# Guardar submission
submission <- data.frame(
  property_id = test_final$property_id,
  price = test_final$pred_rf
)

write.csv(submission, "submission_rf.csv", row.names = FALSE)
