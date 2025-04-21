suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(arrow))

# Set directory and file structure
umap_dir <- file.path("../../4.analyze_data/notebooks/UMAP/results")
plate3_name <- "localhost230405150001"
umap_file <- paste0("UMAP_", plate3_name, "_sc_feature_selected.tsv.gz")
umap_path <- file.path(umap_dir, umap_file)

output_fig_dir <- file.path("figures")

# Define output figure paths
output_umap_file <- file.path(
    output_fig_dir,
    paste0("UMAP_", plate3_name)
)

print(output_umap_file)

# Load in the umap data
umap_df <- read.csv(umap_path, sep = "\t")

cell_count_df <- umap_df %>%
    dplyr::group_by(Metadata_Well) %>%
    dplyr::count()

umap_cp_df <- umap_df %>%
    dplyr::left_join(cell_count_df, by = "Metadata_Well")

# Print shape and head for UMAP dataframe
dim(umap_cp_df)
head(umap_cp_df)

# Read the normalized Parquet file and select the required columns
actin_feature_df <- read_parquet("../../3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_normalized.parquet") %>%
  select(Metadata_Well, Metadata_Site, Metadata_Nuclei_Number_Object_Number, Cells_Intensity_IntegratedIntensityEdge_Actin)

# Read the annotated Parquet file and select the required columns
neighbors_feature_df <- read_parquet("../../3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_annotated.parquet") %>%
  select(Metadata_Well, Metadata_Site, Metadata_Nuclei_Number_Object_Number, Nuclei_Neighbors_NumberOfNeighbors_Adjacent)

# Read the annotated Parquet file and select the required columns
nuclei_area_feature_df <- read_parquet("../../3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_annotated.parquet") %>%
  select(Metadata_Well, Metadata_Site, Metadata_Nuclei_Number_Object_Number, Nuclei_AreaShape_Area)

# Display the filtered dataframe
dim(actin_feature_df)
head(actin_feature_df)
dim(neighbors_feature_df)
head(neighbors_feature_df)
dim(nuclei_area_feature_df)
head(nuclei_area_feature_df)

# Ensure only the specific metadata columns are used for the join
metadata_cols <- c("Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number")

# Ensure these Metadata columns are character type, lowercase, and trimmed of whitespace in all dataframes
actin_feature_df <- actin_feature_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

neighbors_feature_df <- neighbors_feature_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

umap_cp_df <- umap_cp_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

# Ensure these Metadata columns are character type, lowercase, and trimmed of whitespace in all dataframes
nuclei_area_feature_df <- nuclei_area_feature_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

# Perform the join using the selected "Metadata" columns for all three feature dataframes
combined_df <- umap_cp_df %>%
  left_join(actin_feature_df, by = metadata_cols) %>%
  left_join(neighbors_feature_df, by = metadata_cols) %>%
  left_join(nuclei_area_feature_df, by = metadata_cols)

# Remove columns ending with '.y' (duplicate columns from the join) and rename columns ending with '.x'
combined_df <- combined_df %>%
  select(-ends_with(".y")) %>%  # Remove columns ending with '.y'
  rename_with(~ gsub("\\.x$", "", .), ends_with(".x"))  # Rename columns ending with '.x' by removing the suffix

# Display the joined dataframe
dim(combined_df)
head(combined_df)

# Filter the combined_df to exclude "drug_x" treatment (not included in manuscript)
# and remove TGFRi + Healthy (not included in original manuscript UMAP)
combined_df <- combined_df %>%
    dplyr::filter(
        Metadata_treatment != "drug_x" &
        !(Metadata_treatment == "TGFRi" & Metadata_cell_type == "healthy")
    )

# Display the dataframe
dim(combined_df)
head(combined_df)

# Generating UMAPs 
options(repr.plot.width = 12, repr.plot.height = 10)  # Adjust width and height as desired
output_file <- paste0(output_umap_file, "_top_actin_feature.png")

int_umap <- ggplot(combined_df, aes(x = UMAP0, y = UMAP1, color = Cells_Intensity_IntegratedIntensityEdge_Actin)) +
  geom_point(alpha = 0.3) +
  scale_color_viridis_c(
    option = "magma", 
    limits = c(min(combined_df$Cells_Intensity_IntegratedIntensityEdge_Actin), 
               max(combined_df$Cells_Intensity_IntegratedIntensityEdge_Actin)),
    oob = scales::squish
  ) +
  labs(x = "UMAP0", y = "UMAP1", color = "Cells Actin\nIntensity") +
  theme_bw() +
  theme(
    legend.position = "right",
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    plot.title = element_text(size = 20, hjust = 0.5)
  )

# Save the plot
ggsave(output_file, int_umap, dpi = 500, height = 10, width = 12)

int_umap

# Generating UMAPs 
options(repr.plot.width = 12, repr.plot.height = 10)  # Adjust width and height as desired
output_file <- paste0(output_umap_file, "_nuclei_neighbors.png")

neighbors_umap <- ggplot(combined_df, aes(x = UMAP0, y = UMAP1, color = Nuclei_Neighbors_NumberOfNeighbors_Adjacent)) +
  geom_point(alpha = 0.3) +
  scale_color_viridis_c(
    option = "viridis", 
  ) +
  labs(x = "UMAP0", y = "UMAP1", color = "Number of\nNuclei Neighbors") +
  theme_bw() +
  theme(
    legend.position = "right",
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    plot.title = element_text(size = 20, hjust = 0.5)
  )

# Save the plot
ggsave(output_file, neighbors_umap, dpi = 500, height = 10, width = 12)

neighbors_umap


# Generating UMAPs
options(repr.plot.width = 12, repr.plot.height = 10)  # Adjust width and height as desired
output_file <- paste0(output_umap_file, "_area_nuclei.png")

area_nuclei_umap <- ggplot(combined_df, aes(x = UMAP0, y = UMAP1, color = Nuclei_AreaShape_Area)) +
  geom_point(alpha = 0.8) +
  
  # Option 4: Using distiller (color palette based on continuous data)
  scale_color_distiller(palette = "PuBu", direction = 1) +
  
  labs(x = "UMAP0", y = "UMAP1", color = "Nuclei Area") +
  theme_bw() +
  theme(
    legend.position = "right",
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    plot.title = element_text(size = 20, hjust = 0.5)
  )

# Save the plot
ggsave(output_file, area_nuclei_umap, dpi = 500, height = 10, width = 12)

area_nuclei_umap
