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

# Display the filtered dataframe
dim(actin_feature_df)
head(actin_feature_df)

# Ensure only the specific metadata columns are used for the join
metadata_cols <- c("Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number")

# Ensure these Metadata columns are character type, lowercase, and trimmed of whitespace in both dataframes
actin_feature_df <- actin_feature_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

umap_cp_df <- umap_cp_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

# Perform the join using the selected "Metadata" columns
combined_df <- umap_cp_df %>%
  left_join(actin_feature_df, by = metadata_cols)

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

umap <- ggplot(combined_df, aes(x = UMAP0, y = UMAP1, color = Cells_Intensity_IntegratedIntensityEdge_Actin)) +
  geom_point(alpha = 0.3) +  # Set alpha for transparency
  scale_color_gradientn(
    colors = c("#FF9C00", "#FF6100", "#FF4500", "#D32F2F", "#C62828", "#B71C1C"), 
    limits = c(min(combined_df$Cells_Intensity_IntegratedIntensityEdge_Actin), 
               max(combined_df$Cells_Intensity_IntegratedIntensityEdge_Actin)),  # Set custom limits
    oob = scales::squish  # Ensure out-of-bound values are squished into the limits
  ) +  # Color gradient for continuous data
  labs(x = "UMAP0", y = "UMAP1", color = "Cells Actin\nIntensity") +
  theme_bw() +
  theme(
    legend.position = "right",  # Position of the legend
    axis.title = element_text(size = 18),  # Increase axis title font size
    axis.text = element_text(size = 16),  # Increase axis text font size
    legend.title = element_text(size = 16),  # Increase legend title font size
    legend.text = element_text(size = 14),  # Increase legend text font size
    plot.title = element_text(size = 20, hjust = 0.5)  # Optional: center-align title with larger size
  )

# Save the plot
ggsave(output_file, umap, dpi = 500, height = 10, width = 12)

umap
