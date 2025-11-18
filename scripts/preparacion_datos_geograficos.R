# ============================
# 1. PAQUETES
# ============================
library(sf)
library(dplyr)
library(nngeo)
library(units)

# ============================
# 2. RUTAS (AJUSTA A TU CARPETA)
# ============================
ruta <- "/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3"

# ============================
# 3. CARGA DE CAPAS SELECCIONADAS
# ============================
areaactividad      <- st_read(file.path(ruta, "areaactividad.gpkg"))
cicloinfra         <- st_read(file.path(ruta, "cicloinfraestructura.gpkg"))
ecoprin            <- st_read(file.path(ruta, "estructuraecologicaprincipal.gpkg"))
edu_sup            <- st_read(file.path(ruta, "ecosistema_educacion_superior.gpkg"))
centros_comerc     <- st_read(file.path(ruta, "gran_centro_comercial.gpkg"))
IRLoc              <- st_read(file.path(ruta, "IRLoc.gpkg"))
IRSCAT             <- st_read(file.path(ruta, "IRSICAT.gpkg"))
parques            <- st_read(file.path(ruta, "parque.gpkg"))
red_vial           <- st_read(file.path(ruta, "redinfraestructuravialarterial.gpkg"))
uso_res            <- st_read(file.path(ruta, "sector_uso_residencial.gpkg"))

# ============================
# 4. CARGA DE LA CAPA OBJETIVO
# (PUNTOS O POLÍGONOS A PREDECIR)
# ============================
objetivo <- st_read(file.path(ruta, "objetivo.gpkg"))

# Asegurar que todo esté en el mismo CRS
crs_target <- st_crs(objetivo)

capas <- list(areaactividad, cicloinfra, ecoprin, edu_sup,
              centros_comerc, IRLoc, IRSCAT,
              parques, red_vial, uso_res)

capas <- lapply(capas, st_transform, crs = crs_target)

# reasignar
areaactividad      <- capas[[1]]
cicloinfra         <- capas[[2]]
ecoprin            <- capas[[3]]
edu_sup            <- capas[[4]]
centros_comerc     <- capas[[5]]
IRLoc              <- capas[[6]]
IRSCAT             <- capas[[7]]
parques            <- capas[[8]]
red_vial           <- capas[[9]]
uso_res            <- capas[[10]]

# ============================
# 5. FUNCIÓN PARA DISTANCIAS
# ============================
distancia_a <- function(obj, capa) {
  st_nn(obj, capa, k = 1, progress = FALSE, returnDist = TRUE)$dist %>% 
    set_units("m") %>% drop_units()
}

# ============================
# 6. GENERACIÓN DE VARIABLES
# ============================

df <- objetivo %>% 
  mutate(
    
    # -------- DISTANCIAS --------
    dist_red_vial = distancia_a(., red_vial),
    dist_cicloinfra = distancia_a(., cicloinfra),
    dist_ecoprin = distancia_a(., ecoprin),
    dist_edu_sup = distancia_a(., edu_sup),
    dist_centros_comerc = distancia_a(., centros_comerc),
    dist_parque = distancia_a(., parques),
    
    # -------- PERTENENCIA --------
    dentro_area_act = as.integer(st_intersects(., areaactividad, sparse = FALSE)),
    dentro_uso_res = as.integer(st_intersects(., uso_res, sparse = FALSE)),
    
    # -------- ATRIBUTOS DIRECTOS --------
    categoria_IRLoc = IRLoc$CATEGORIA[ st_nearest_feature(., IRLoc) ],
    categoria_IRSCAT = IRSCAT$CATEGORIA[ st_nearest_feature(., IRSCAT) ]
  )

# ============================
# 7. GUARDAR LA BD FINAL
# ============================
st_write(df, file.path(ruta, "dataset_prediccion.gpkg"), delete_dsn = TRUE)

