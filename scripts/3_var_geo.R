###############################################################
#   GEOSPATIAL FEATURE ENGINEERING & MODELING PIPELINE        #
#   Author: Andrez Guerrero                                   #
#   Description:                                              #
#   This script loads residential data, reads key geospatial  #
#   layers of Bogotá, removes Sumapaz, computes spatial       #
#   distances & contextual variables, trains a Random Forest  #
#   model, and exports final predictions.                     #
###############################################################

# -------------------------
# 1. Load required packages
# -------------------------

library(sf)
library(dplyr)
library(purrr)
library(caret)
library(randomForest)

# -----------------------------------
# 2. Load training data and convert to sf
# -----------------------------------

train <- read.csv("train.csv")

train_sf <- st_as_sf(
  train,
  coords = c("lon", "lat"),
  crs = 4326
) |>
  st_transform(3116)   # MAGNA-SIRGAS Bogotá

# ----------------------------------------------------------
# 3. List of selected GPKG files (clean and usable only)
# ----------------------------------------------------------

archivos_utiles <- c(
  "areaactividad.gpkg",
  "cicloinfraestructura.gpkg",
  "ecosistema_educacion_superior.gpkg",
  "gran_centro_comercial.gpkg",
  "IRLoc.gpkg",
  "IRUPZ.gpkg",
  "parque.gpkg",
  "redinfraestructuravialarterial.gpkg",
  "sector_uso_residencial.gpkg"
)

# ----------------------------------------------------------
# 4. Function to safely read and clean each geospatial layer
# ----------------------------------------------------------

leer_capa <- function(path) {
  
  nm <- st_layers(path)$name[1]
  cat("Loading layer:", nm, "\n")
  
  sf <- st_read(path, layer = nm, quiet = TRUE)
  sf <- st_make_valid(sf)
  sf <- st_transform(sf, 3116)
  
  sf$capa <- nm
  return(sf)
}

# ------------------------------------------
# 5. Read all usable geospatial layers as sf
# ------------------------------------------

lista_sf <- map(archivos_utiles, leer_capa)

# ------------------------------------------------------------
# 6. Remove geometries belonging to Sumapaz (southern Bogotá)
# ------------------------------------------------------------

sup_lat <- 4.5  # Approximate latitude threshold

lista_sf <- map(lista_sf, function(x) {
  x <- x[st_coordinates(st_centroid(x))[, 2] > sup_lat, ]
  x
})

# -------------------------------------------
# 7. Combine all layers for visualization (optional)
# -------------------------------------------

todo_sf <- bind_rows(lista_sf)

# -------------------------------------------------
# 8. Extract specific layers needed for distances
# -------------------------------------------------

parques <- lista_sf[[ which(sapply(lista_sf, \(x) "Parque" %in% x$capa)) ]]
ciclo   <- lista_sf[[ which(sapply(lista_sf, \(x) "Red_cicloinfraestructura" %in% x$capa)) ]]
cc      <- lista_sf[[ which(sapply(lista_sf, \(x) "Gran_centro_comercial" %in% x$capa)) ]]
upz     <- lista_sf[[ which(sapply(lista_sf, \(x) "IRUPZ" %in% x$capa)) ]]
loc     <- lista_sf[[ which(sapply(lista_sf, \(x) "IRLoc" %in% x$capa)) ]]
uso     <- lista_sf[[ which(sapply(lista_sf, \(x) "Sector_uso_residencial" %in% x$capa)) ]]

# ---------------------------------------
# 9. Compute distance-based variables
# ---------------------------------------

train_sf$dist_parque <- apply(st_distance(train_sf, parques), 1, min)
train_sf$dist_ciclo  <- apply(st_distance(train_sf, ciclo), 1, min)
train_sf$dist_cc     <- apply(st_distance(train_sf, cc), 1, min)

# ---------------------------------------
# 10. (Optional) Join polygon attributes
# ---------------------------------------

# train_sf <- st_join(train_sf, upz["UPZ"])
# train_sf <- st_join(train_sf, loc["LOCALIDAD"])
# train_sf <- st_join(train_sf, uso["UsoTUso"])

# ---------------------------------------
# 11. Buffer-based density indicators
# ---------------------------------------

train_sf$buffer300 <- st_buffer(train_sf, 300)

train_sf$parques_300m <- lengths(st_intersects(train_sf$buffer300, parques))
train_sf$ciclo_300m   <- lengths(st_intersects(train_sf$buffer300, ciclo))

# -------------------------------
# 12. Final modeling dataset
# -------------------------------

train_model <- train_sf |>
  st_drop_geometry() |>
  select(-buffer300)

write.csv(train_model, "train_model.csv", row.names = FALSE)

cat("\n✓ Geospatial feature engineering completed.\n")
cat("✓ Output saved as train_model.csv\n\n")

###############################################################
#           RANDOM FOREST MODELING SECTION                    #
###############################################################

train_model <- read.csv("train_model.csv")

target <- "price"
set.seed(123)

# Data split
train_index <- createDataPartition(train_model[[target]], p = 0.8, list = FALSE)
train_data  <- train_model[train_index, ]
test_data   <- train_model[-train_index, ]

# Remove missing values
train_clean <- na.omit(train_data)

# Train Random Forest
rf_model <- randomForest(
  price ~ .,
  data = train_clean,
  ntree = 500,
  mtry = 6,
  importance = TRUE
)

# Predictions
pred_rf <- predict(rf_model, newdata = test_data)

# Performance metrics
rmse <- sqrt(mean((pred_rf - test_data[[target]])^2))
mae  <- mean(abs(pred_rf - test_data[[target]]))
r2   <- cor(pred_rf, test_data[[target]])^2

cat("\nModel performance:\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R2:", r2, "\n")

varImpPlot(rf_model)

# Export predictions
results <- test_data |>
  mutate(pred_rf = pred_rf)

write.csv(results, "rf_ntree500_mtry6_v1.csv", row.names = FALSE)

cat("\n✓ Predictions exported as rf_ntree500_mtry6_v1.csv\n")
