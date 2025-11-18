###############################################################
#   GEOSPATIAL FEATURE ENGINEERING (solo parques, ciclo, cc)  #
###############################################################

library(sf)
library(dplyr)
library(purrr)
library(randomForest)

# ---------------------------------------------------
# 1. Load train & test data
# ---------------------------------------------------

train <- read.csv("train.csv")
test  <- read.csv("test.csv")

train_sf <- st_as_sf(train, coords = c("lon","lat"), crs = 4326) |> 
  st_transform(3116)

test_sf <- st_as_sf(test, coords = c("lon","lat"), crs = 4326) |> 
  st_transform(3116)

# ----------------------------------------------------------
# 2. Read ONLY required GPKG layers
# ----------------------------------------------------------

archivos_utiles <- c(
  "parque.gpkg",
  "cicloinfraestructura.gpkg",
  "gran_centro_comercial.gpkg"
)

leer_capa <- function(path) {
  nm <- st_layers(path)$name[1]
  cat("Loading layer:", nm, "\n")
  
  sf <- st_read(path, layer = nm, quiet=TRUE)
  sf <- st_make_valid(sf)
  sf <- st_transform(sf, 3116)
  sf$capa <- nm
  sf
}

lista_sf <- map(archivos_utiles, leer_capa)

# ------------------------------------------------------------
# 3. Remove Sumapaz (fast filter by centroid Y coordinate)
# ------------------------------------------------------------

sup_lat_y3116 <- 1020000  # approx Y of lat 4.5° in EPSG 3116

lista_sf <- map(lista_sf, \(x) {
  x[ st_coordinates(st_centroid(x))[,2] > sup_lat_y3116, ]
})

# -------------------------------------------------
# 4. Extract layers
# -------------------------------------------------

get_layer <- function(name){
  lista_sf[[ which(sapply(lista_sf, \(x) name %in% x$capa)) ]]
}

parques <- get_layer("Parque")
ciclo   <- get_layer("Red_cicloinfraestructura")
cc      <- get_layer("Gran_centro_comercial")

# -------------------------------------------------
# 5. Compute features (optimized, memory-safe)
# -------------------------------------------------

compute_features <- function(pts){
  
  message("→ Computing nearest distances...")
  
  # Nearest index per layer
  idx_parques <- st_nearest_feature(pts, parques)
  idx_ciclo   <- st_nearest_feature(pts, ciclo)
  idx_cc      <- st_nearest_feature(pts, cc)
  
  # Distances (element-wise, tiny memory footprint)
  pts$dist_parque <- st_distance(pts, parques[idx_parques,], by_element = TRUE)
  pts$dist_ciclo  <- st_distance(pts, ciclo[idx_ciclo,],    by_element = TRUE)
  pts$dist_cc     <- st_distance(pts, cc[idx_cc,],          by_element = TRUE)
  
  message("→ Counting features within 300 meters...")
  
  pts$parques_300m <- lengths(st_is_within_distance(pts, parques, dist = 300))
  pts$ciclo_300m   <- lengths(st_is_within_distance(pts, ciclo,   dist = 300))
  pts$cc_300m      <- lengths(st_is_within_distance(pts, cc,       dist = 300))
  
  pts |> st_drop_geometry()
}

# -------------------------------------------------
# 6. Build features
# -------------------------------------------------

cat("\nComputing geospatial features for TRAIN...\n")
train_model <- compute_features(train_sf)

cat("\nComputing geospatial features for TEST...\n")
test_model  <- compute_features(test_sf)

# Same columns
common_cols <- intersect(names(train_model), names(test_model))
train_model <- train_model[, common_cols]
test_model  <- test_model[, common_cols]

train_model$price <- train$price

write.csv(train_model, "train_model_minimal.csv", row.names=FALSE)

# -------------------------------------------------
# 7. Train Random Forest
# -------------------------------------------------

set.seed(123)
train_clean <- na.omit(train_model)

rf_model <- randomForest(
  price ~ .,
  data = train_clean,
  ntree = 500,
  mtry = 6,
  importance = TRUE
)

cat("\nRandom Forest trained.\n")

# -------------------------------------------------
# 8. Predict test
# -------------------------------------------------

pred_test <- predict(rf_model, newdata = test_model)

submission <- data.frame(
  property_id = test$property_id,
  price = pred_test
)

write.csv(submission, 
          "submission_rf_minimal.csv",
          row.names = FALSE)

cat("\n✓ Submission file saved as submission_rf_minimal.csv\n")

