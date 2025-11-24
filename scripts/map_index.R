############################################################
# LIBRARIES
############################################################
library(sf)
library(dplyr)
library(ggplot2)
library(viridis)
library(readr)
library(patchwork)

############################################################
# 1. LOAD MAIN DATASET
############################################################

base_path <- "/Users/andrezconz/Library/Mobile Documents/com~apple~CloudDocs/MECA/BD&ML/Set_3/uniandes-bdml-2025-20-ps-3"
csv_path  <- file.path(base_path, "train_ready_geo.csv")

df <- read_csv(csv_path, show_col_types = FALSE)

# Convert to sf
df_sf <- st_as_sf(df, coords = c("lon", "lat"), crs = 4326)

# Extract lon/lat back for ggplot
df_sf <- df_sf %>%
  mutate(
    lon = st_coordinates(.)[,1],
    lat = st_coordinates(.)[,2]
  )

############################################################
# 2. LOAD BOGOTÁ GEOMETRY (IRLoc)
############################################################

gpkg_path <- file.path(base_path, "IRLoc.gpkg")
bogota <- st_read(gpkg_path, quiet = TRUE)
bogota <- st_transform(bogota, 4326)

############################################################
# 3. CLIP BOGOTÁ (REMOVE SUMAPAZ)
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
# 4. GENERAL MAPPING FUNCTION FOR URBAN INDICES
############################################################

plot_urban_index <- function(df, bg, var, title) {
  ggplot() +
    
    # Light Bogotá background
    geom_sf(
      data = bg,
      fill = "#F7F9FB",      # very light blue-grey
      color = "#D0D7DD",     # subtle outlines
      size = 0.25
    ) +
    
    # Points colored by index
    geom_point(
      data = df,
      aes(x = lon, y = lat, color = .data[[var]]),
      size = 1.4,
      alpha = 0.85
    ) +
    
    # Professional, CLEAR color palette
    scale_color_viridis_c(
      option = "cividis",    # MUCH clearer and professional than mako/plasma
      direction = 1
    ) +
    
    coord_sf(expand = FALSE) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid = element_blank(),
      legend.position = "right",
      plot.title = element_text(face = "bold", size = 16)
    ) +
    labs(
      title = title,
      color = var
    )
}

############################################################
# 5. GENERATE THE THREE MAPS
############################################################

p_ICUR <- plot_urban_index(
  df_sf_clip, bogota_clip, "ICUR",
  "Urban Residential Complexity (ICUR)"
)

p_CMP <- plot_urban_index(
  df_sf_clip, bogota_clip, "CMP",
  "Metropolitan Centrality (CMP)"
)

p_NUR <- plot_urban_index(
  df_sf_clip, bogota_clip, "NUR",
  "Urban Residential Interaction (NUR)"
)

############################################################
# 6. COMBINE INTO A SINGLE FIGURE
############################################################

final_plot <- (p_ICUR | p_CMP | p_NUR)

final_plot
