# =====================================================================================
# Geospatial feature engineering - Versión corregida y lista para ejecutar
# =====================================================================================

library(sf)
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)

cat("\n--- GEOSPATIAL: START ---\n")

# ----------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------
proj_target <- 3116         # proyección métrica para Bogotá
lat_threshold <- 4.5        # filtro por latitud si no hay límite administrativo
distances_m <- c(300, 500)  # radios de interés

# directorio de trabajo (ajusta si hace falta)
setwd('/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3/uniandes-bdml-2025-20-ps-3')

# ==========================================================
# A) Archivos geopackage a cargar (local + /mnt/data fallback)
# ==========================================================
geopkgs <- c(
  "parque.gpkg",
  "cicloinfraestructura.gpkg",
  "gran_centro_comercial.gpkg",
  "eatu.gpkg",
  "estacionpolicia.gpkg",
  "colegios06_2025.gpkg",
  "IRLoc.gpkg"
)

# ----------------------------------------------------------
# B) helper: leer layer primero disponible dentro del gpkg
# ----------------------------------------------------------
read_layer <- function(path, target_crs = proj_target) {
  if (!file.exists(path)) stop("Archivo no encontrado: ", path)
  layers <- st_layers(path)$name
  # usar la primera capa por defecto
  layer_name <- layers[1]
  message("Loading: ", path, " -> layer: ", layer_name)
  x <- st_read(path, layer = layer_name, quiet = TRUE)
  x <- st_make_valid(x)
  # transform solo si tiene CRS; st_transform tolera si ya está
  x <- st_transform(x, target_crs)
  # anotar nombre de la capa original (para nombrar la lista)
  attr(x, "layer_name") <- layer_name
  return(x)
}

# ----------------------------------------------------------
# C) Cargar todas las capas (silencioso pero con mensajes)
# ----------------------------------------------------------
lista_sf <- map(geopkgs, safely(read_layer))
# filtrar exitosos
lista_ok <- keep(lista_sf, ~ is.null(.x$error))
lista_sf <- map(lista_ok, "result")

# crear nombres robustos: preferimos el layer_name si existe, sino el basename
names(lista_sf) <- map_chr(lista_sf, function(x) {
  ln <- attr(x, "layer_name")
  if (!is.null(ln) && nzchar(ln)) return(ln)
  tools::file_path_sans_ext(basename(attr(x, "source")))
})

cat("Capas cargadas:", paste(names(lista_sf), collapse = ", "), "\n")

# acceso seguro
get_layer <- function(name) {
  if (!name %in% names(lista_sf)) stop("Layer not found: ", name, "\nDisponibles: ",
                                       paste(names(lista_sf), collapse = ", "))
  lista_sf[[name]]
}

# ----------------------------------------------------------
# D) Leer train/test y convertir a sf
# ----------------------------------------------------------
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test  <- read.csv("test.csv", stringsAsFactors = FALSE)

# verificar columnas lon/lat
if (!all(c("lon", "lat") %in% names(train))) {
  stop("train debe tener columnas 'lon' y 'lat'")
}
if (!all(c("lon", "lat") %in% names(test))) {
  stop("test debe tener columnas 'lon' y 'lat'")
}

# chequear unicidad de property_id en raw
stopifnot(!any(duplicated(train$property_id)), !any(duplicated(test$property_id)))

train_sf <- st_as_sf(train, coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>% st_transform(proj_target)
test_sf  <- st_as_sf(test,  coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>% st_transform(proj_target)

# ----------------------------------------------------------
# E) Extraer capas que usaremos (nombres exactos detectados)
# ----------------------------------------------------------
# Ajusta estos nombres si en tu sistema difieren; uso los nombres que observaste.
parques <- get_layer("Parque")
ciclo   <- get_layer("Red_cicloinfraestructura")
cc      <- get_layer("Gran_centro_comercial")
eatu    <- get_layer("EATu")
policia <- get_layer("EstacionPolicia")
colegios <- {
  # preferir la capa que venga bajo ese nombre o usar la del /mnt/data si existe
  nm <- intersect(names(lista_sf), c("colegios06_2025", "colegios06_2025.gpkg", "colegios06_2025"))
  if (length(nm)) get_layer(nm[1]) else NULL
}
# IRLoc (localidades)
loc     <- get_layer("IRLoc")

# Asegurar que todas las capas relevantes estén en CRS objetivo
to_transform <- list(parques, ciclo, cc, eatu, policia, colegios, loc)
to_transform <- keep(to_transform, ~ !is.null(.x))
to_transform <- lapply(to_transform, function(x) st_transform(x, proj_target))

# reasignar (orden fijo)
parques <- to_transform[[1]]
ciclo   <- to_transform[[2]]
cc      <- to_transform[[3]]
eatu    <- to_transform[[4]]
policia <- to_transform[[5]]
if (length(to_transform) >= 6) colegios <- to_transform[[6]]
if (length(to_transform) >= 7) loc <- to_transform[[7]]

# ----------------------------------------------------------
# F) Funciones utilitarias (distancias y conteos)
# ----------------------------------------------------------
dist_to_nearest <- function(x_points, y_features) {
  # devuelve numeric en metros
  idx <- st_nearest_feature(x_points, y_features)
  d <- st_distance(x_points, y_features[idx, ], by_element = TRUE)
  as.numeric(d)
}

count_within <- function(x_points, y_features, dist_m) {
  mat <- st_is_within_distance(x_points, y_features, dist = dist_m)
  lengths(mat)
}

# ----------------------------------------------------------
# G) compute_geo_features: incluir nuevas amenities ESPECÍFICAS
# ----------------------------------------------------------
compute_geo_features <- function(sf_points) {
  # espera sf con CRS proyectado
  if (!inherits(sf_points, "sf")) stop("sf_points debe ser un objeto sf")
  # distancias
  sf_points$dist_parque <- dist_to_nearest(sf_points, parques)
  sf_points$dist_ciclo  <- dist_to_nearest(sf_points, ciclo)
  sf_points$dist_cc     <- dist_to_nearest(sf_points, cc)
  sf_points$dist_eatu   <- dist_to_nearest(sf_points, eatu)
  sf_points$dist_policia<- dist_to_nearest(sf_points, policia)
  if (!is.null(colegios)) {
    sf_points$dist_colegio <- dist_to_nearest(sf_points, colegios)
  }
  # counts
  sf_points$parques_300m <- count_within(sf_points, parques, 300)
  sf_points$ciclo_300m   <- count_within(sf_points, ciclo, 300)
  sf_points$cc_300m      <- count_within(sf_points, cc, 300)
  sf_points$eatu_300m    <- count_within(sf_points, eatu, 300)
  if (!is.null(colegios)) {
    sf_points$colegios_300m <- count_within(sf_points, colegios, 300)
    sf_points$colegios_500m <- count_within(sf_points, colegios, 500)
  }
  # 500m counts (solo para las principales)
  sf_points$parques_500m <- count_within(sf_points, parques, 500)
  sf_points$ciclo_500m   <- count_within(sf_points, ciclo, 500)
  sf_points$cc_500m      <- count_within(sf_points, cc, 500)
  sf_points$eatu_500m    <- count_within(sf_points, eatu, 500)
  # densidades (conteo / área de circulo)
  area_300 <- pi * (300^2)
  area_500 <- pi * (500^2)
  sf_points <- sf_points %>%
    mutate(
      dens_parques_300 = parques_300m / area_300,
      dens_ciclo_300 = ciclo_300m / area_300,
      dens_cc_300 = cc_300m / area_300,
      dens_eatu_300 = eatu_300m / area_300,
      dens_parques_500 = parques_500m / area_500,
      dens_ciclo_500 = ciclo_500m / area_500,
      dens_cc_500 = cc_500m / area_500,
      dens_eatu_500 = eatu_500m / area_500
    )
  # ratios estables
  sf_points <- sf_points %>%
    mutate(
      ratio_parque = if_else(dist_parque == 0, parques_300m + 1, parques_300m / (dist_parque + 1)),
      ratio_ciclo  = if_else(dist_ciclo  == 0, ciclo_300m + 1,   ciclo_300m  / (dist_ciclo  + 1)),
      ratio_cc     = if_else(dist_cc     == 0, cc_300m + 1,      cc_300m     / (dist_cc     + 1)),
      ratio_eatu   = if_else(dist_eatu   == 0, eatu_300m + 1,    eatu_300m   / (dist_eatu   + 1))
    )
  # coords proyectadas y polinomios
  coords <- st_coordinates(sf_points)
  sf_points$x <- coords[,1]
  sf_points$y <- coords[,2]
  sf_points <- sf_points %>% mutate(x2 = x^2, y2 = y^2, xy = x * y)
  # drop geom and select
  df <- sf_points %>%
    st_drop_geometry() %>%
    select(property_id, starts_with("dist_"), starts_with("parques_"), starts_with("ciclo_"),
           starts_with("cc_"), starts_with("eatu_"), starts_with("colegios_"),
           starts_with("dens_"), starts_with("ratio_"),
           x, y, x2, y2, xy)
  return(df)
}

# ----------------------------------------------------------
# H) Calcular features espaciales para train & test
# ----------------------------------------------------------
cat("\nComputing geospatial features for TRAIN (optimized)…\n")
train_geo <- compute_geo_features(train_sf)

cat("\nComputing geospatial features for TEST (optimized)…\n")
test_geo  <- compute_geo_features(test_sf)

# ============================
# PCA de amenities (robusto)
# ============================

amen_cols <- intersect(amen_cols, names(train_geo))

# imputación median
amen_train <- train_geo[, amen_cols] %>% 
  mutate_all(~ ifelse(is.na(.), median(., na.rm=TRUE), .))

# eliminar columnas con varianza cero
zero_var_cols <- names(amen_train)[apply(amen_train, 2, sd, na.rm=TRUE) == 0]

if (length(zero_var_cols) > 0) {
  message("Columnas con varianza cero eliminadas del PCA: ", paste(zero_var_cols, collapse=", "))
  amen_train <- amen_train[, !(names(amen_train) %in% zero_var_cols)]
}

# re-definir amen_cols sin esas columnas
amen_cols_pca <- names(amen_train)

# escalar
scaling_center <- apply(amen_train, 2, median, na.rm=TRUE)
scaling_scale  <- apply(amen_train, 2, sd,     na.rm=TRUE)

amen_train_scaled <- scale(amen_train, center = scaling_center, scale = scaling_scale)

# ============================
# PCA de amenities (robusto)
# ============================

amen_cols <- intersect(amen_cols, names(train_geo))

# imputación median
amen_train <- train_geo[, amen_cols] %>% 
  mutate_all(~ ifelse(is.na(.), median(., na.rm=TRUE), .))

# eliminar columnas con varianza cero
zero_var_cols <- names(amen_train)[apply(amen_train, 2, sd, na.rm=TRUE) == 0]

if (length(zero_var_cols) > 0) {
  message("Columnas con varianza cero eliminadas del PCA: ", paste(zero_var_cols, collapse=", "))
  amen_train <- amen_train[, !(names(amen_train) %in% zero_var_cols)]
}

# re-definir amen_cols sin esas columnas
amen_cols_pca <- names(amen_train)

# escalar
scaling_center <- apply(amen_train, 2, median, na.rm=TRUE)
scaling_scale  <- apply(amen_train, 2, sd,     na.rm=TRUE)

amen_train_scaled <- scale(amen_train, center = scaling_center, scale = scaling_scale)

# PCA
pca <- prcomp(amen_train_scaled)

# agregar componentes
train_geo <- bind_cols(
  train_geo,
  as.data.frame(pca$x[,1:3]) %>% rename_with(~paste0("amen_pca",1:3))
)

# ============================
# PCA para TEST
# ============================

amen_test <- test_geo[, amen_cols] %>% 
  mutate_all(~ ifelse(is.na(.), median(., na.rm=TRUE), .))

# quitar mismas columnas que en train
amen_test <- amen_test[, amen_cols_pca]

amen_test_scaled <- scale(amen_test, center = scaling_center, scale = scaling_scale)

pcs_test <- as.matrix(amen_test_scaled) %*% pca$rotation[,1:3]

test_geo <- bind_cols(
  test_geo,
  as.data.frame(pcs_test) %>% rename_with(~paste0("amen_pca",1:3))
)



# ----------------------------------------------------------
# J) ICUR / CMP / NUR
# ----------------------------------------------------------
compute_ICUR <- function(df) {
  df$ICUR <-
    scale(df$dens_parques_300) +
    scale(df$dens_ciclo_300) +
    scale(df$dens_cc_300) +
    scale(df$dens_eatu_300)
  df
}
compute_CMP <- function(df) {
  df$CMP <- 1 / (df$dist_parque + df$dist_ciclo + df$dist_cc + df$dist_eatu + 1)
  df
}

train_geo <- compute_ICUR(train_geo)
test_geo  <- compute_ICUR(test_geo)
train_geo <- compute_CMP(train_geo)
test_geo  <- compute_CMP(test_geo)
train_geo$NUR <- train_geo$ICUR * train_geo$CMP
test_geo$NUR  <- test_geo$ICUR * test_geo$CMP

# ----------------------------------------------------------
# K) Unir variables de localidad (IRLoc) y crear tasas/crime features
# ----------------------------------------------------------
# aplicar función de creación de features sobre loc (sf)
create_crime_features <- function(loc_sf) {
  # renombrar nombre de localidad
  if ("CMNOMLOCAL" %in% names(loc_sf)) {
    loc_sf <- loc_sf %>% rename(localidad = CMNOMLOCAL)
  } else if ("CMIULOCAL" %in% names(loc_sf)) {
    loc_sf <- loc_sf %>% rename(localidad = CMIULOCAL)
  } else {
    stop("No se encontró campo de localidad en IRLoc")
  }
  
  # helper
  safe_get <- function(df, col) if (col %in% names(df)) df[[col]] else 0
  
  # crear hurtos_totales (suma de los totales más relevantes)
  loc_sf$hurtos_totales <- safe_get(loc_sf, "CMRTOTAL") +
    safe_get(loc_sf, "CMNTOTAL") + safe_get(loc_sf, "CMMTOTAL") +
    safe_get(loc_sf, "CMHCTOTAL") + safe_get(loc_sf, "CMDTOTAL")
  
  # tasa por km2
  if ("SHAPE_AREA" %in% names(loc_sf)) {
    loc_sf$tasa_hurto_km2 <- loc_sf$hurtos_totales / (loc_sf$SHAPE_AREA / 1e6)
  } else {
    loc_sf$tasa_hurto_km2 <- NA_real_
  }
  
  # violencia total
  loc_sf$violencia_total <- safe_get(loc_sf, "CMAOPTOTAL") +
    safe_get(loc_sf, "CMMMTOTAL") + safe_get(loc_sf, "CMPIATOTAL")
  
  # indice general y binario
  loc_sf$delito_total_general <- loc_sf$hurtos_totales + loc_sf$violencia_total
  loc_sf$var_hurto <- safe_get(loc_sf, "CMRVAR")
  loc_sf$zona_segura <- ifelse(loc_sf$tasa_hurto_km2 < median(loc_sf$tasa_hurto_km2, na.rm = TRUE), 1, 0)
  
  return(loc_sf)
}

# aplicar a loc (sf)
loc_sf <- create_crime_features(loc)  # loc ya en proj_target
# hacer join espacial: train_geo/test_geo son data.frames; convertimos temporalmente a sf
train_geo_sf <- st_as_sf(train_geo, coords = c("x","y"), crs = proj_target, remove = FALSE)
test_geo_sf  <- st_as_sf(test_geo,  coords = c("x","y"), crs = proj_target, remove = FALSE)

# join espacial y dropear geometría de polygons (conservar columnas de loc)
join_cols <- c("localidad", "hurtos_totales", "tasa_hurto_km2",
               "violencia_total", "delito_total_general", "var_hurto", "zona_segura")

train_geo_sf <- st_join(train_geo_sf, loc_sf %>% select(all_of(join_cols)))
test_geo_sf  <- st_join(test_geo_sf,  loc_sf %>% select(all_of(join_cols)))

train_geo <- st_drop_geometry(train_geo_sf)
test_geo  <- st_drop_geometry(test_geo_sf)

# ----------------------------------------------------------
# L) Índice compuesto ICRIM (normalizar tasas y promediar)
# ----------------------------------------------------------
cols_tasas <- grep("^tas_", names(train_geo), value = TRUE)
# si no hay tas_... calculadas, usamos tasa_hurto_km2
if (length(cols_tasas) == 0 && "tasa_hurto_km2" %in% names(train_geo)) {
  cols_tasas <- "tasa_hurto_km2"
}

if (length(cols_tasas) > 0) {
  train_geo$ICRIM <- rowMeans(scale(train_geo[, cols_tasas]), na.rm = TRUE)
  test_geo$ICRIM  <- rowMeans(scale(test_geo[, cols_tasas]),  na.rm = TRUE)
} else {
  train_geo$ICRIM <- NA_real_
  test_geo$ICRIM  <- NA_real_
}

# ----------------------------------------------------------
# M) Merge final con datos originales y guardar
# ----------------------------------------------------------
train_final <- left_join(train_geo, train, by = "property_id")
test_final  <- left_join(test_geo,  test,  by = "property_id")

stopifnot(nrow(train_final) == nrow(train))
stopifnot(nrow(test_final)  == nrow(test))
names(train_final)
write.csv(train_final, "train_final_geo.csv", row.names = FALSE)
write.csv(test_final,  "test_final_geo.csv",  row.names = FALSE)

cat("\n✓ Saved: train_ready_geo.csv & test_ready_geo.csv\n")
