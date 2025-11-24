# =======================================================
# PS3 — Chapinero Prices
# Neural Network (Keras)
# =======================================================

# -------------------------------------------------------
# 1) Load libraries
# -------------------------------------------------------
# pacman automatically installs missing packages and loads them
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,
    tidymodels,    # recipes, workflows
    keras,         # deep learning API
    reticulate     # virtualenv management for keras backend
  )
})

tidymodels_prefer()   # Avoid API conflicts

# Use the virtual environment where Keras + TensorFlow were installed
reticulate::use_virtualenv("r-reticulate", required = TRUE)

# -------------------------------------------------------
# 2) Load training and test datasets
# -------------------------------------------------------
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

# Integrity checks
stopifnot("property_id" %in% names(train))
stopifnot("price"       %in% names(train))

# Ensure numeric target
train$price <- as.numeric(train$price)

# -------------------------------------------------------
# 3) Preprocessing recipe for the neural network
# -------------------------------------------------------
# Neural networks require normalized inputs and cannot handle factors directly.
# This preprocessing pipeline:
#  - Removes property_id
#  - Handles novel categories
#  - Imputes missing data
#  - Converts categorical variables into dummy variables
#  - Normalizes all numeric predictors (critical for Keras)

rec_nn <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%      # Exclude ID column
  step_novel(all_nominal_predictors()) %>%           # New factor levels in test set
  step_impute_median(all_numeric_predictors()) %>%   # Median imputation
  step_impute_mode(all_nominal_predictors()) %>%     # Mode imputation
  step_dummy(all_nominal_predictors()) %>%           # One-hot encoding
  step_normalize(all_numeric_predictors(), -all_outcomes())

# Prepare recipe
prep_nn <- prep(rec_nn, training = train)

# Bake train and test datasets
train_baked <- bake(prep_nn, new_data = train)
test_baked  <- bake(prep_nn, new_data = test)

# -------------------------------------------------------
# 4) Build model matrices for Keras
# -------------------------------------------------------
x_cols <- setdiff(names(train_baked), "price")

X_train <- as.matrix(train_baked[, x_cols, drop = FALSE])
y_train <- as.numeric(train_baked$price)
X_test  <- as.matrix(test_baked[, x_cols, drop = FALSE])

# Mean of target — used to initialize output layer bias
y_mean <- mean(y_train, na.rm = TRUE)

cat("Train matrix dim:", dim(X_train), "\n")
cat("Test matrix dim :", dim(X_test),  "\n")
cat("Mean target     :", y_mean, "\n")

# -------------------------------------------------------
# 5) Define neural network architecture
# -------------------------------------------------------
k_clear_session()   # Reset previous computation graphs

n_inputs <- ncol(X_train)

# Fully connected feed-forward network with two hidden layers.
# He-normal initialization is used because we apply ReLU activations.
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
    units            = 1,
    activation       = "linear",
    bias_initializer = initializer_constant(y_mean)
  )

summary(model)

# -------------------------------------------------------
# 6) Compile the model
# -------------------------------------------------------
# MSE as loss, Adam optimizer, and MAE for interpretability.
model %>% compile(
  loss      = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics   = list("mean_absolute_error")
)

# -------------------------------------------------------
# 7) Train the network
# -------------------------------------------------------
set.seed(2025)

history <- model %>% fit(
  x                = X_train,
  y                = y_train,
  epochs           = 300,         # Increase if training curve is improving
  batch_size       = 128,
  validation_split = 0.20,        # 20% of training used as validation
  verbose          = 1
)

# Plot learning curves
plot(history)

# -------------------------------------------------------
# 8) Evaluate training performance (optional)
# -------------------------------------------------------
eval_train <- model %>% evaluate(X_train, y_train, verbose = 0)
cat("Train MSE:", eval_train["loss"], "\n")
cat("Train MAE:", eval_train["mean_absolute_error"], "\n")

# -------------------------------------------------------
# 9) Predict on the TEST dataset
# -------------------------------------------------------
pred_test_nn <- model %>% predict(X_test)
pred_test_nn <- as.numeric(pred_test_nn)

# Round DOWN to nearest COP million (Kaggle style)
pred_test_nn_round <- floor(pred_test_nn / 1e6) * 1e6

submission_nn <- tibble(
  property_id = test$property_id,
  price       = pred_test_nn_round
)

head(submission_nn)

# -------------------------------------------------------
# 10) Save Kaggle submission
# -------------------------------------------------------
output_file <- "NN_Keras_spatial_prepro.csv"

write.csv(
  submission_nn,
  output_file,
  row.names = FALSE
)

cat("Submission saved as:", output_file, "\n")

# -------------------------------------------------------
# End of script
# -------------------------------------------------------
