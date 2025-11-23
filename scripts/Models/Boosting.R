# =======================================================
# PS3 — Chapinero Prices
# XGBoost
# =======================================================

# ================
# 0) Load packages
# ================

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,    # dplyr, ggplot2, readr, etc.
    tidymodels,   # recipes, parsnip, workflows, tuning, metrics
    sf,           # spatial objects (st_as_sf)
    spatialsample,# spatial_block_cv for spatial CV
    xgboost,      # gradient boosting engine
    doParallel,   # parallel backend for tidymodels tuning
    ggplot2,      # plotting
    forcats,      # factor reordering (fct_reorder)
    foreach       # for foreach backend registration
  )
})

tidymodels_prefer()   # avoid conflicts between packages

# ==============
# 1) Load data 
# ==============
train <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/train_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

test <- read.csv(
  "C:/Users/Sergio/Documents/Problem-Set-3-Making-Money-with-ML2/data/test_unified_final.csv",
  na.strings = c("", "NA", "NaN")
)

# Drop property_type if present (not useful as is, and you did this elsewhere)
if ("property_type" %in% names(train)) train <- train %>% dplyr::select(-property_type)
if ("property_type" %in% names(test))  test  <- test  %>% dplyr::select(-property_type)

# Basic checks: spatial coordinates must exist
stopifnot(all(c("lon","lat") %in% names(train)))

# Target must be numeric for XGBoost
train$price <- as.numeric(train$price)

# ===========================
# 2) Create spatial CV folds 
# ===========================
set.seed(2025)

# Convert train into an sf object using lon/lat as geometry
train_sf <- sf::st_as_sf(
  train,
  coords = c("lon","lat"),
  crs    = 4326,
  remove = FALSE
)

# Create 5 spatial blocks (spatial cross-validation)
block_folds <- spatialsample::spatial_block_cv(train_sf, v = 5)

# ======================================================
# 3) Recipe: preprocessing pipeline for XGBoost
# ======================================================
# This replicates what you already used:
# - Treat property_id as an ID (not a predictor)
# - Handle novel factor levels in test
# - Impute numeric variables with median
# - Impute categorical variables with mode
# - Create dummy variables for categorical predictors
rec_xgb <- recipe(price ~ ., data = train) %>%
  update_role(property_id, new_role = "id") %>%      # do not use property_id as a feature
  step_novel(all_nominal_predictors()) %>%           # handle unseen factor levels in test
  step_impute_median(all_numeric_predictors()) %>%   # numeric imputation
  step_impute_mode(all_nominal_predictors()) %>%     # categorical imputation
  step_dummy(all_nominal_predictors())               # one-hot encoding for factors

# ======================================================
# 4) XGBoost model specification with tuning
# ======================================================
# p = number of original predictors (before dummy expansion)
p <- length(setdiff(names(train), c("price", "property_id")))

xgb_spec <- boost_tree(
  trees          = 650,             # number of trees (fixed from your best model)
  mtry           = floor(sqrt(p)),  # number of variables tried at each split
  min_n          = tune(),          # hyperparameter to tune
  tree_depth     = tune(),          # hyperparameter to tune
  learn_rate     = tune(),          # hyperparameter to tune
  sample_size    = tune(),          # hyperparameter to tune
  loss_reduction = 0                # gamma (here fixed at 0)
) %>%
  set_mode("regression") %>%
  set_engine("xgboost", objective = "reg:squarederror")

# Combine recipe + model into a workflow
wf_xgb <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rec_xgb)

# ======================================================
# 5) Hyperparameter grid for tuning
# ======================================================
# This is the same grid you used before:
xgb_grid <- tidyr::crossing(
  min_n       = c(5, 20),
  tree_depth  = c(4, 7, 10),
  learn_rate  = c(0.03, 0.06),
  sample_size = c(0.7, 0.9)
)

# ======================================================
# 6) Tuning with spatial CV in parallel
# ======================================================
# We parallelize over resamples (spatial folds)
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1

cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)

control_xgb <- control_grid(
  parallel_over = "resamples",  # each fold in parallel
  verbose       = TRUE
)

set.seed(2025)
xgb_tuned <- tune_grid(
  wf_xgb,
  resamples = block_folds,        # spatial folds
  grid      = xgb_grid,           # hyperparameter combinations
  metrics   = metric_set(mae),    # we optimize MAE
  control   = control_xgb
)

# Stop the cluster and return to sequential
parallel::stopCluster(cl)
foreach::registerDoSEQ()

# Collect tuning metrics (MAE per hyperparameter combination)
xgb_metrics <- collect_metrics(xgb_tuned)
print(xgb_metrics)

# ======================================================
# 7) Extract best hyperparameters and fit final model
# ======================================================
# Best combination according to mean MAE
xgb_best <- xgb_metrics %>%
  dplyr::filter(.metric == "mae") %>%
  dplyr::arrange(mean) %>%
  dplyr::slice(1)

print(xgb_best)

# Keep only the tuning parameters
best_params <- xgb_best %>%
  dplyr::select(min_n, tree_depth, learn_rate, sample_size)

# Final workflow with best hyperparameters plugged in
wf_xgb_final <- finalize_workflow(wf_xgb, best_params)

# Fit final XGBoost model on the full training set
fit_xgb <- fit(wf_xgb_final, data = train)

# ==============================
# 8) Variable importance (Gain) 
# ==============================
# Extract the underlying xgboost model (xgb.Booster)
xgb_obj <- fit_xgb %>%
  extract_fit_engine()

# Compute importance (Gain, Cover, etc.)
xgb_imp <- xgboost::xgb.importance(model = xgb_obj)

# Convert Gain to percentage of total Gain
xgb_imp$GainPct <- 100 * xgb_imp$Gain / sum(xgb_imp$Gain)

# Dictionary: map field names to clear academic labels (Spanish)
nice_names <- c(
  "bathrooms"        = "Baños",
  "surface_total"    = "Área total (m²)",
  "surface_covered"  = "Área construida (m²)",
  "rooms"            = "Habitaciones",
  "bedrooms"         = "Dormitorios",
  "parqueaderos"     = "Parqueaderos",
  "lat"              = "Latitud",
  "lon"              = "Longitud",
  "dist_parque"      = "Distancia al parque más cercano (m)",
  "dist_ciclo"       = "Distancia a la ciclorruta más cercana (m)",
  "dist_cc"          = "Distancia al centro comercial más cercano (m)",
  "dist_eatu"        = "Distancia a establecimientos de alojamiento y turismo (EATU) (m)",
  "dist_policia"     = "Distancia a la estación de policía más cercana (m)",
  "dist_colegio"     = "Distancia al colegio más cercano (m)",
  "parques_300m"     = "Parques en un radio de 300 m",
  "parques_500m"     = "Parques en un radio de 500 m",
  "ciclo_300m"       = "Cicloinfraestructura en un radio de 300 m",
  "ciclo_500m"       = "Cicloinfraestructura en un radio de 500 m",
  "cc_300m"          = "Centros comerciales en un radio de 300 m",
  "cc_500m"          = "Centros comerciales en un radio de 500 m",
  "eatu_300m"        = "Establecimientos de alojamiento y turismo (EATU) en 300 m",
  "eatu_500m"        = "Establecimientos de alojamiento y turismo (EATU) en 500 m",
  "ratio_parque"     = "Accesibilidad relativa a parques",
  "ratio_ciclo"      = "Accesibilidad relativa a ciclorrutas",
  "ratio_cc"         = "Accesibilidad relativa a centros comerciales",
  "ratio_eatu"       = "Accesibilidad relativa a EATU",
  "ICUR"             = "Índice de Complejidad Urbana Residencial (ICUR)",
  "CMP"              = "Centralidad Metropolitana Proximal (CMP)",
  "NUR"              = "Interacción ICUR × CMP (NUR)",
  "ICRIM"            = "Índice compuesto de criminalidad"
)

# Apply pretty labels when available, otherwise keep the original name
xgb_imp$FeatureNice <- ifelse(
  xgb_imp$Feature %in% names(nice_names),
  nice_names[xgb_imp$Feature],
  xgb_imp$Feature
)

# Keep the TOP 15 most important variables (by Gain %)
xgb_imp_top <- xgb_imp %>%
  dplyr::arrange(desc(GainPct)) %>%
  dplyr::slice_head(n = 15)

# Print top 15 in console
cat("\n=== TOP 15 variables by Gain (%) ===\n")
print(xgb_imp_top[, c("Feature", "FeatureNice", "GainPct")])

# Save full importance table to CSV
write.csv(xgb_imp, "xgb_importancia_variables_full.csv", row.names = FALSE)

# ======================================================
# 9) Plot variable importance and export figure
# ======================================================
g <- ggplot(xgb_imp_top,
            aes(x = GainPct,
                y = forcats::fct_reorder(FeatureNice, GainPct))) +
  geom_col(fill = "#1f77b4") +
  geom_text(aes(label = sprintf("%.1f%%", GainPct)),
            hjust = -0.1, size = 4) +
  labs(
    title    = "Variable importance in XGBoost (Gain %)",
    subtitle = "Gain = relative contribution to error reduction of the model",
    x        = "Importance (% of total Gain)",
    y        = "Variable"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title    = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 13),
    axis.text.y   = element_text(size = 11),
    axis.text.x   = element_text(size = 11)
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)))

# Show plot in RStudio
print(g)

# Export PNG and PDF
ggsave("xgb_importancia_variables.png", plot = g, width = 10, height = 7, dpi = 300)
ggsave("xgb_importancia_variables.pdf", plot = g, width = 10, height = 7)

cat("\nGraphs saved as: xgb_importancia_variables.png and .pdf\n")

# ======================================================
# 10) Predictions on test set and Kaggle submission file
# ======================================================
pred_test <- predict(fit_xgb, new_data = test)$.pred

# Round predictions down to the nearest million 
pred_test_round <- floor(pred_test / 1e6) * 1e6

submission <- tibble(
  property_id = test$property_id,
  price       = pred_test_round
)

# Name of submission file includes tuned hyperparameters
fname <- sprintf(
  "XGB_tm_minn%s_depth%s_lr%s_ss%s_with_importance.csv",
  best_params$min_n,
  best_params$tree_depth,
  best_params$learn_rate,
  best_params$sample_size
)

write.csv(submission, fname, row.names = FALSE)

cat("\n=== DONE: XGBoost tuned + variable importance + plots + submission ===\n")
