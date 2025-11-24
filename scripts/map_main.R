############################################################
# Libraries
############################################################
library(sf)
library(dplyr)
library(purrr)
library(readr)
library(ggplot2)

############################################################
# 1. Base Path of the Project
############################################################

base_path <- "/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3/uniandes-bdml-2025-20-ps-3"

############################################################
# 2. Load Processed Geospatial CSV
############################################################

csv_path <- file.path(base_path, "train_ready_geo.csv")

df <- read_csv(csv_path, show_col_types = FALSE)

# Convert to sf (lon/lat, WGS84)
df_sf <- st_as_sf(df, coords = c("lon", "lat"), crs = 4326)

# Transform to Bogotá CRS (EPSG 3116)
df_sf <- st_transform(df_sf, 3116)

############################################################
# 3. List of GPKG files to load
############################################################

gpkg_names <- c(
  "parque.gpkg",
  "cicloinfraestructura.gpkg",
  "gran_centro_comercial.gpkg",
  "eatu.gpkg",
  "estacionpolicia.gpkg",
  "colegios06_2025.gpkg",
  "IRLoc.gpkg"
)

gpkg_paths <- file.path(base_path, gpkg_names)

############################################################
# 4. Safe loader: read layer + transform to 3116
############################################################

read_layer_safe <- function(path) {
  layer <- st_read(path, quiet = TRUE)
  st_transform(layer, 3116)
}

############################################################
# 5. Load all layers into a named list
############################################################

layers <- map(gpkg_paths, read_layer_safe)
names(layers) <- tools::file_path_sans_ext(gpkg_names)

print("Loaded layers:")
print(names(layers))

############################################################
# 6. Convert everything to EPSG 4326 before clipping
############################################################

layers_4326 <- lapply(layers, function(x) {
  if (st_crs(x)$epsg != 4326) st_transform(x, 4326) else x
})

df_sf_4326 <- st_transform(df_sf, 4326)

############################################################
# 7. Clip to Bogotá bounding box (excluding Sumapaz)
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

IRLoc_clip      <- st_crop(layers_4326$IRLoc, bbox_bogota)
parques_clip    <- st_crop(layers_4326$parque, bbox_bogota)
ciclo_clip      <- st_crop(layers_4326$cicloinfraestructura, bbox_bogota)
cc_clip         <- st_crop(layers_4326$gran_centro_comercial, bbox_bogota)
eatu_clip       <- st_crop(layers_4326$eatu, bbox_bogota)
policia_clip    <- st_crop(layers_4326$estacionpolicia, bbox_bogota)
df_sf_clip      <- st_crop(df_sf_4326, bbox_bogota)

############################################################
# 8. Final Map — Clean, professional, no legends
############################################################

ggplot() +
  
  # --- Background (IRLoc polygons) ---
  geom_sf(
    data = IRLoc_clip,
    fill = "#F4F6F7",       # very light grey background
    color = "#AAB7C4",      # subtle grey border
    size = 0.15
  ) +
  
  # --- Parks ---
  geom_sf(
    data = parques_clip,
    fill = "#A1CDA8",
    color = NA,
    alpha = 0.45
  ) +
  
  # --- Cycling infrastructure ---
  geom_sf(
    data = ciclo_clip,
    color = "#4A6FA5",
    size = 0.35,
    alpha = 0.8
  ) +
  
  # --- Shopping centers ---
  geom_sf(
    data = cc_clip,
    color = "#E7A85B",
    size = 1.5,
    alpha = 0.9
  ) +
  
  # --- EATU (education/culture nodes) ---
  geom_sf(
    data = eatu_clip,
    color = "#7C6F9A",
    size = 1.2,
    alpha = 0.85
  ) +
  
  # --- Police stations ---
  geom_sf(
    data = policia_clip,
    color = "#C94C4C",
    size = 1.2,
    alpha = 0.9
  ) +
  
  # --- Style ---
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  
  # --- Labels ---
  labs(
    title = "Geospatial Urban Layers — Bogotá",
    subtitle = "Parks, cycling network, malls, police.",
    x = NULL,
    y = NULL
  )
