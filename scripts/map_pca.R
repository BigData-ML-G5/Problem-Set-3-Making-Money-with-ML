############################################################
# LIBRARÍAS
############################################################
library(sf)
library(dplyr)
library(ggplot2)
library(viridis)
library(readr)
library(patchwork)

############################################################
# 1. CARGAR DATASET PROCESADO
############################################################

base_path <- "/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3/uniandes-bdml-2025-20-ps-3"
csv_path  <- file.path(base_path, "train_ready_geo.csv")

df <- read_csv(csv_path, show_col_types = FALSE)

# Convertir a sf
df_sf <- st_as_sf(df, coords = c("lon", "lat"), crs = 4326)

# Extraer lon/lat nuevamente para ggplot
df_sf <- df_sf %>%
  mutate(
    lon = st_coordinates(.)[,1],
    lat = st_coordinates(.)[,2]
  )

############################################################
# 2. CARGAR POLÍGONO BOGOTÁ (IRLoc)
############################################################

gpkg_path <- file.path(base_path, "IRLoc.gpkg")
bogota <- st_read(gpkg_path, quiet = TRUE)
bogota <- st_transform(bogota, 4326)

############################################################
# 3. RECORTAR BOGOTÁ SIN SUMAPAZ
############################################################

bbox_bogota <- st_bbox(
  c(
    xmin = -74.20,
    xmax = -73.00,
    ymin =  4.55,
    ymax =  4.78
  ),
  crs = st_crs(4326)
)

bogota_clip <- st_crop(bogota, bbox_bogota)
df_sf_clip  <- st_crop(df_sf, bbox_bogota)

############################################################
# 4. FUNCIÓN PARA MAPEAR PCA (ESTILO PRO)
############################################################

plot_pca <- function(df, bg, var, title) {
  ggplot() +
    
    # --- Fondo Bogotá ---
    geom_sf(
      data = bg,
      fill = "#F3F6F9",      # fondo gris-azulado muy claro
      color = "#C9D3DB",     # bordes sutiles
      size = 0.22
    ) +
    
    # --- Puntos coloreados por el PCA ---
    geom_point(
      data = df,
      aes(x = lon, y = lat, color = .data[[var]]),
      size = 1.4,
      alpha = 0.85
    ) +
    
    # --- Paleta clara y profesional ---
    scale_color_viridis_c(
      option = "cividis",    # clara, legible, profesional
      direction = 1
    ) +
    
    coord_sf(expand = FALSE) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "right",
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", size = 16)
    ) +
    
    labs(
      title = title,
      color = var
    )
}

############################################################
# 5. GENERAR MAPAS PCA
############################################################

p1 <- plot_pca(df_sf_clip, bogota_clip, "amen_pca1",
               "PCA1 — General Amenity Density")

p2 <- plot_pca(df_sf_clip, bogota_clip, "amen_pca2",
               "PCA2 — Cycle vs. Commercial Structure")

p3 <- plot_pca(df_sf_clip, bogota_clip, "amen_pca3",
               "PCA3 — Tourism / Cultural Intensity")

############################################################
# 6. UNIR EN UNA SOLA FIGURA
############################################################

final_plot <- (p1 | p2 | p3)

final_plot
