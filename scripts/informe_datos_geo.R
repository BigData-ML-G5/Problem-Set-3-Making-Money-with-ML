library(sf)
library(dplyr)
library(purrr)
library(stringr)

# ------------------------------------------
# 1. Ruta donde están tus archivos GPKG
# ------------------------------------------
ruta <- "/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3"

setwd(ruta)

# ------------------------------------------
# 2. Listar archivos .gpkg
# ------------------------------------------
archivos <- list.files(ruta, pattern = "\\.gpkg$", full.names = TRUE)

message("Archivos encontrados:")
print(basename(archivos))

# ------------------------------------------
# 3. Función para inspeccionar un GPKG
# ------------------------------------------
inspeccionar_gpkg <- function(path) {
  
  capas <- st_layers(path)$name
  
  map_df(capas, function(capa) {
    message(paste0("Procesando capa: ", capa, " en archivo: ", basename(path)))
    
    # Leer capa
    sf_obj <- tryCatch(
      st_read(path, layer = capa, quiet = TRUE),
      error = function(e) NULL
    )
    
    # Si falla la lectura
    if (is.null(sf_obj)) {
      return(tibble(
        archivo = basename(path),
        capa = capa,
        geometria = NA,
        n_filas = NA,
        n_cols = NA,
        columnas = NA,
        sugerencia = "ERROR: no se pudo cargar esta capa"
      ))
    }
    
    # Extraer información
    tipo_geo <- class(sf_obj$geom)[1]
    cols <- colnames(sf_obj)
    
    # Reglas inteligentes para modelos
    sugerencia <- case_when(
      tipo_geo %in% c("sfc_POINT", "sfc_MULTIPOINT") ~ 
        "Útil para variables de distancia: parques, SITP, comercios, universidades.",
      
      tipo_geo %in% c("sfc_LINESTRING", "sfc_MULTILINESTRING") ~ 
        "Útil para distancia a redes: vías, ciclorutas, troncales, ríos.",
      
      tipo_geo %in% c("sfc_POLYGON", "sfc_MULTIPOLYGON") ~ 
        "Útil para pertenencia zonal: UPL, barrios, manzanas, usos del suelo.",
      
      TRUE ~ "Tipo de geometría desconocido."
    )
    
    tibble(
      archivo = basename(path),
      capa = capa,
      geometria = tipo_geo,
      n_filas = nrow(sf_obj),
      n_cols = length(cols),
      columnas = paste(cols, collapse = ", "),
      sugerencia = sugerencia
    )
  })
}

# ------------------------------------------
# 4. Ejecutar inspección para TODOS los GPKG
# ------------------------------------------
informe <- map_df(archivos, inspeccionar_gpkg)

# ------------------------------------------
# 5. Mostrar en pantalla
# ------------------------------------------
print(informe)

# ------------------------------------------
# 6. Guardar informe
# ------------------------------------------

write.csv(informe, "informe_capas_gpkg.csv", row.names = FALSE)

message(">>> Informe generado correctamente: informe_capas_gpkg.csv")
