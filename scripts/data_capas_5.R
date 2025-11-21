# =====================================================================================
# Geospatial feature engineering - Reescritura optimizada
# - Usa st_nearest_feature + st_distance(by_element=TRUE) para distancias eficientes
# - Usa st_is_within_distance para conteos en radio (evita crear buffers explícitos)
# - Filtra SUMAPAZ en CRS 4326 (latitud) correctamente
# - Crea densidades, ratios, PCA sobre amenities y polinomios espaciales
# - Verifica unicidad de property_id antes de joins
# =====================================================================================

library(sf)
library(dplyr)
library(purrr)
library(tidyr)
library(RANN)        # opcional, no usado si usamos st_nearest_feature
library(ggplot2)
# nngeo no es necesario aquí; usamos sf::st_nearest_feature y st_is_within_distance

cat("\n--- GEOSPATIAL: START ---\n")

# ----------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------
proj_target <- 3116         # proyección para cálculos métricos (Bogotá)
lat_threshold <- 4.5        # para eliminar Sumapaz en CRS 4326
distances_m <- c(300, 500)  # radios de interés

# ----------------------------------------------------------
# 1. Read train/test (as data.frames) and convert to sf
#    (asumes train and test are lists or data.frames already cargadas)
# ----------------------------------------------------------
# bind if train/test son listas; si ya son data.frames, adaptarlo
train_all <- bind_rows(train)    # tu objeto original
test_all  <- bind_rows(test)

# chequear unicidad de property_id
stopifnot(!any(duplicated(train_all$property_id)))
stopifnot(!any(duplicated(test_all$property_id)))

# convertir a sf (WGS84) y luego proyectar a métrico
train_sf <- st_as_sf(train_all, coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(proj_target)
test_sf  <- st_as_sf(test_all,  coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(proj_target)

# ----------------------------------------------------------
# 2. Cargar geopackages (lista nombrada robusta)
# ----------------------------------------------------------
setwd('/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3/uniandes-bdml-2025-20-ps-3')

geopkgs <- c(
  "parque.gpkg",
  "cicloinfraestructura.gpkg",
  "gran_centro_comercial.gpkg",
  "eatu.gpkg",
  "estacionpolicia.gpkg"
)

read_layer <- function(path) {
  nm <- st_layers(path)$name[1]
  message("Loading: ", path, " -> layer: ", nm)
  sf <- st_read(path, layer = nm, quiet = TRUE)
  sf <- st_make_valid(sf)
  sf <- st_transform(sf, proj_target)
  sf$capa <- nm
  return(sf)
}

lista_sf <- map(geopkgs, read_layer)
# nombrar por capa (primer valor único de $capa)
names(lista_sf) <- map_chr(lista_sf, ~ unique(.x$capa)[1])

# acceso seguro (da error si no existe)
get_layer <- function(name) {
  if (!name %in% names(lista_sf)) stop("Layer not found: ", name)
  lista_sf[[name]]
}

# ----------------------------------------------------------
# 3. FILTRAR SUMAPAZ correctamente (usar CRS 4326 para latitud)
# ----------------------------------------------------------
# creamos una función que recorta cada capa al límite administrativo de Bogotá
# (si tienes un geopackage con límite de Bogotá, es ideal; si no, usamos latitud)
# Recomendado: usar 'limite_bogota.gpkg' si lo tienes. Aquí fallback por latitud.

names(lista_sf)


# Intento de leer límite de Bogotá si existe
if (file.exists("limite_bogota.gpkg")) {
  bog <- st_read("limite_bogota.gpkg", quiet = TRUE) %>% st_transform(proj_target)
  message("Usando limite_bogota.gpkg para recorte de capas.")
  lista_sf <- map(lista_sf, ~ st_intersection(.x, bog))
} else {
  message("No existe limite_bogota.gpkg -> filtrando por latitud (CRS 4326)")
  # función que filtra geometrías cuyo centroide en latitud > lat_threshold
  lista_sf <- map(lista_sf, function(x) {
    cent4326 <- st_transform(st_centroid(x), 4326)
    keep <- st_coordinates(cent4326)[,2] > lat_threshold
    x[keep, ]
  })
}

# ----------------------------------------------------------
# 4. Seleccionar capas (nombres robustos)
# ----------------------------------------------------------
# Ajusta estos nombres a los valores reales de tus capas
parques <- get_layer("Parque")
ciclo   <- get_layer("Red_cicloinfraestructura")
cc      <- get_layer("Gran_centro_comercial")
eatu    <- get_layer("EATu")
policia <- get_layer("EstacionPolicia")


# Para cálculos de "distancia", usar la geometría tal cual (polígono/linea)
# Para conteos por proximidad, st_is_within_distance funciona bien

# ----------------------------------------------------------
# 5. Funciones utilitarias: distancia al feature más cercano (eficiente)
# ----------------------------------------------------------
# st_nearest_feature devuelve índice del feature más cercano en 'y' para cada 'x'
dist_to_nearest <- function(x_points, y_features) {
  idx <- st_nearest_feature(x_points, y_features)          # vector longitud nrow(x_points)
  # calcular distancias por elemento (evita matriz NxM)
  d <- st_distance(x_points, y_features[idx, ], by_element = TRUE)
  # st_distance devuelve units; convertir a numeric (metros)
  as.numeric(d)
}

# conteos dentro de distancia (usa st_is_within_distance -> devuelve lista esparsa)
count_within <- function(x_points, y_features, dist_m) {
  mat <- st_is_within_distance(x_points, y_features, dist = dist_m)
  lengths(mat)
}

# ----------------------------------------------------------
# 6. compute_geo_features optimizada (no buffers persistentes)
# ----------------------------------------------------------
compute_geo_features <- function(sf_points) {
  # distancias a la feature más cercana (eficiente)
  sf_points$dist_parque <- dist_to_nearest(sf_points, parques)
  sf_points$dist_ciclo  <- dist_to_nearest(sf_points, ciclo)
  sf_points$dist_cc     <- dist_to_nearest(sf_points, cc)
  sf_points$dist_eatu   <- dist_to_nearest(sf_points, eatu)
  sf_points$dist_policia<- dist_to_nearest(sf_points, policia)
  
  # counts dentro de radios (sin crear buffers explícitos)
  sf_points$parques_300m <- count_within(sf_points, parques, 300)
  sf_points$ciclo_300m   <- count_within(sf_points, ciclo, 300)
  sf_points$cc_300m      <- count_within(sf_points, cc, 300)
  sf_points$eatu_300m    <- count_within(sf_points, eatu, 300)
  
  sf_points$parques_500m <- count_within(sf_points, parques, 500)
  sf_points$ciclo_500m   <- count_within(sf_points, ciclo, 500)
  sf_points$cc_500m      <- count_within(sf_points, cc, 500)
  sf_points$eatu_500m    <- count_within(sf_points, eatu, 500)
  
  # densidades (conteo / área)
  area_300 <- pi * (300^2)
  area_500 <- pi * (500^2)
  sf_points <- sf_points %>%
    mutate(
      dens_parques_300 = parques_300m / area_300,
      dens_ciclo_300   = ciclo_300m   / area_300,
      dens_cc_300      = cc_300m      / area_300,
      dens_eatu_300    = eatu_300m    / area_300,
      dens_parques_500 = parques_500m / area_500,
      dens_ciclo_500   = ciclo_500m   / area_500,
      dens_cc_500      = cc_500m      / area_500,
      dens_eatu_500    = eatu_500m    / area_500
    )
  
  # ratios útiles (estables entre áreas)
  sf_points <- sf_points %>%
    mutate(
      ratio_parque = if_else(dist_parque == 0, parques_300m + 1, parques_300m / (dist_parque + 1)),
      ratio_ciclo  = if_else(dist_ciclo  == 0, ciclo_300m + 1,   ciclo_300m  / (dist_ciclo  + 1)),
      ratio_cc     = if_else(dist_cc     == 0, cc_300m + 1,      cc_300m     / (dist_cc     + 1)),
      ratio_eatu   = if_else(dist_eatu   == 0, eatu_300m + 1,    eatu_300m   / (dist_eatu   + 1))
    )
  
  # coordenadas proyectadas y polinomios (para que el modelo aprenda el mapa)
  coords <- st_coordinates(sf_points)
  sf_points$x <- coords[,1]
  sf_points$y <- coords[,2]
  sf_points <- sf_points %>%
    mutate(x2 = x^2, y2 = y^2, xy = x * y)
  
  # dropear geometría y retornar df (mantener columnas clave)
  df <- sf_points %>%
    st_drop_geometry() %>%
    select(property_id, starts_with("dist_"), starts_with("parques_"), starts_with("ciclo_"),
           starts_with("cc_"), starts_with("eatu_"), starts_with("dens_"), starts_with("ratio_"),
           x, y, x2, y2, xy)
  
  return(df)
}

cat("\nComputing geospatial features for TRAIN (optimized)…\n")
train_geo <- compute_geo_features(train_sf)

cat("\nComputing geospatial features for TEST (optimized)…\n")
test_geo  <- compute_geo_features(test_sf)

# ----------------------------------------------------------
# 7. PCA / componentes latentes sobre amenities (fit en train, transformar test)
# ----------------------------------------------------------
amen_cols <- c("dist_parque", "dist_ciclo", "dist_cc", "dist_eatu",
               "parques_300m", "ciclo_300m", "cc_300m", "eatu_300m")

# normalizar y hacer PCA
amen_train_mat <- train_geo %>%
  select(any_of(amen_cols)) %>%
  mutate_all(~ ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
  scale(center = TRUE, scale = TRUE)

pca <- prcomp(amen_train_mat, center = FALSE, scale. = FALSE)  # ya escalado

# añadir PC1..PC3 a train_geo
pcs_train <- as.data.frame(pca$x)[, 1:3]
colnames(pcs_train) <- paste0("amen_pca", 1:3)
train_geo <- bind_cols(train_geo, pcs_train)

# transformar test (usar rotación y mismo centro/scale)
amen_test_mat <- test_geo %>%
  select(any_of(amen_cols)) %>%
  mutate_all(~ ifelse(is.na(.), median(., na.rm = TRUE), .))
# centrar y escalar con atributos de train
center <- attr(amen_train_mat, "scaled:center")
scalev <- attr(amen_train_mat, "scaled:scale")
amen_test_scaled <- sweep(amen_test_mat, 2, center, "-")
amen_test_scaled <- sweep(amen_test_scaled, 2, scalev, "/")
pcs_test <- as.matrix(amen_test_scaled) %*% pca$rotation[, 1:3]
pcs_test <- as.data.frame(pcs_test)
colnames(pcs_test) <- paste0("amen_pca", 1:3)
test_geo <- bind_cols(test_geo, pcs_test)

# ==============================================================================
# 7.1. Variables avanzadas de idiosincrasia urbana
# - ICUR: Complejidad urbana (densidades escaladas)
# - CMP: Centralidad metropolitana proximal
# - NUR: Interacción ICUR × CMP
# ==============================================================================

# 1) ICUR = Índice de Complejidad Urbana Relativa
compute_ICUR <- function(df) {
  df$ICUR <- 
    scale(df$dens_parques_300) +
    scale(df$dens_ciclo_300)   +
    scale(df$dens_cc_300)      +
    scale(df$dens_eatu_300)
  df
}

# 2) CMP = Centralidad Metropolitana Proximal
compute_CMP <- function(df) {
  df$CMP <- 1 / (df$dist_parque +
                   df$dist_ciclo  +
                   df$dist_cc     +
                   df$dist_eatu   + 1)
  df
}

# APLICARlas a TRAIN + TEST
train_geo <- compute_ICUR(train_geo)
test_geo  <- compute_ICUR(test_geo)

train_geo <- compute_CMP(train_geo)
test_geo  <- compute_CMP(test_geo)

# 3) NUR = Nivel Urbano Relativo (ICUR × CMP)
train_geo$NUR <- train_geo$ICUR * train_geo$CMP
test_geo$NUR  <- test_geo$ICUR * test_geo$CMP

# ----------------------------------------------------------
# 8. Merge con base original (sin duplicados, by property_id)
# ----------------------------------------------------------
train_final <- left_join(train_geo, train_all, by = "property_id")
test_final  <- left_join(test_geo,  test_all,  by = "property_id")

# verificaciones
stopifnot(nrow(train_final) == nrow(train_all))
stopifnot(nrow(test_final)  == nrow(test_all))

cat("\n✓ TRAIN + TEST merged with geospatial features.\n")
cat("TRAIN:", nrow(train_final), "rows -", ncol(train_final), "columns\n")
cat("TEST :", nrow(test_final),  "rows -", ncol(test_final),  "columns\n")

# ----------------------------------------------------------
# 9. Guardar resultados
# ----------------------------------------------------------
write.csv(train_final, "train_ready_geo.csv", row.names = FALSE)
write.csv(test_final,  "test_ready_geo.csv",  row.names = FALSE)
cat("\n✓ Saved: train_ready_geo.csv & test_ready_geo.csv\n")


