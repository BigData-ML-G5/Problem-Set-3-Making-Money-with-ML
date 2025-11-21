# =======================================================
# PS3 — Chapinero Prices
# Neural Network (Keras) + Preprocesamiento con recipes
# =======================================================

# ------------------------
# 0) Cargar librerías
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,   # recipes, etc.
    keras,
    reticulate
  )
})

# Usar el entorno virtual donde instalaste keras
reticulate::use_virtualenv("r-reticulate", required = TRUE)

tidymodels_prefer()

# ------------------------
# 1) Cargar datos
# ------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_train_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

test  <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML/data/data_test_text_finished.csv",
  na.strings = c("", "NA", "NaN")
)

# Chequeos básicos
stopifnot("property_id" %in% names(train))
stopifnot("property_id" %in% names(test))
stopifnot("price"       %in% names(train))

# Aseguramos que price sea numérico
train$price <- as.numeric(train$price)

# ------------------------
# 2) Receta de preprocesamiento para la red
#    (similar a la de LM, pero incluyendo normalización)
# ------------------------
rec_nn <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>% # no usar como predictor
  # manejar niveles nuevos en test para variables categóricas
  step_novel(all_nominal_predictors()) %>%
  # imputación
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # dummies para categóricas
  step_dummy(all_nominal_predictors()) %>%
  # normalizar predictores numéricos (muy importante para Keras)
  step_normalize(all_numeric_predictors(), -all_outcomes())

# Preparar receta con train
prep_nn <- prep(rec_nn, training = train)

# "Hornear" train y test
train_baked <- bake(prep_nn, new_data = train)
test_baked  <- bake(prep_nn, new_data = test)

# ------------------------
# 3) Construir matrices X, y para Keras
# ------------------------
# Columnas usadas como predictores (todas menos price)
x_cols <- setdiff(names(train_baked), "price")

X_train <- as.matrix(train_baked[, x_cols, drop = FALSE])
y_train <- as.numeric(train_baked$price)

X_test  <- as.matrix(test_baked[, x_cols, drop = FALSE])

# Media del target (para inicializar el sesgo de la capa de salida)
y_mean <- mean(y_train, na.rm = TRUE)

cat("Dimensiones X_train:", dim(X_train), "\n")
cat("Dimensiones X_test :", dim(X_test), "\n")
cat("Media de y_train   :", y_mean, "\n")

# ------------------------
# 4) Definir la arquitectura de la red neuronal
# ------------------------
k_clear_session()  # limpiar sesiones anteriores, por si acaso

n_inputs <- ncol(X_train)

model <- keras_model_sequential() %>%
  layer_dense(
    units              = 30,
    activation         = "relu",
    input_shape        = n_inputs,
    kernel_initializer = initializer_he_normal()
  ) %>%
  layer_dense(
    units              = 30,
    activation         = "relu",
    kernel_initializer = initializer_he_normal()
  ) %>%
  layer_dense(
    units              = 1,
    activation         = "linear",
    bias_initializer   = initializer_constant(y_mean)
  )

summary(model)

# ------------------------
# 5) Compilar el modelo
# ------------------------
model %>% compile(
  loss      = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics   = list("mean_absolute_error")
)

# ------------------------
# 6) Entrenamiento
# ------------------------
set.seed(2025)

history <- model %>% fit(
  x              = X_train,
  y              = y_train,
  epochs         = 300,      # puedes aumentar si ves que sigue mejorando
  batch_size     = 128,
  validation_split = 0.2,
  verbose        = 1
)

# (Opcional) ver curva de entrenamiento
plot(history)

# ------------------------
# 7) (Opcional) Evaluación en el propio train
#    (para tener una idea de MAE in-sample)
# ------------------------
eval_train <- model %>% evaluate(X_train, y_train, verbose = 0)
cat("MSE train:", eval_train["loss"], "\n")
cat("MAE train:", eval_train["mean_absolute_error"], "\n")

# ------------------------
# 8) Predicciones sobre TEST y archivo para Kaggle
# ------------------------
pred_test_nn <- model %>% predict(X_test)
pred_test_nn <- as.numeric(pred_test_nn)

# Redondeo hacia abajo al millón más cercano
pred_test_nn_round <- floor(pred_test_nn / 1e6) * 1e6

submission_nn <- tibble(
  property_id = test$property_id,
  price       = pred_test_nn_round
)

print(head(submission_nn))

# Nombre del archivo
model_label <- "NN_keras"
fname_nn <- paste0(model_label, "_baseline.csv")

write.csv(submission_nn, fname_nn, row.names = FALSE)
cat("Archivo de submission guardado como:", fname_nn, "\n")

