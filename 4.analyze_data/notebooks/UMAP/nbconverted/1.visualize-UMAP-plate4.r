suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(arrow))

# Set directory and file structure
umap_dir <- file.path("results")
plate4_name <- "localhost231120090001"
umap_file <- "UMAP_localhost231120090001_sc_feature_selected.tsv.gz"
umap_path <- file.path(umap_dir, umap_file)


output_fig_dir <- file.path("figures")
umap_prefix <- "UMAP_"

# Define output figure path
output_umap_file <- file.path(
    output_fig_dir,
    paste0(umap_prefix, plate4_name)
)

print(output_umap_file)

# Load in the umap data
df <- readr::read_tsv(
    umap_path,
    col_types = readr::cols(
        .default = "c",
        "UMAP0" = "d",
        "UMAP1" = "d"
    )
)
cell_count_df <- df %>%
    dplyr::group_by(Metadata_Well) %>%
    dplyr::count() %>%
    dplyr::rename(Metadata_Cell_Count = n)

umap_cp_df <- df %>%
    dplyr::left_join(cell_count_df, by = "Metadata_Well")

head(umap_cp_df)

# Create UMAP labelled with the anomaly score as gradient
umap_heart_num_figure <- 
  ggplot(umap_cp_df, aes(x = UMAP0, y = UMAP1, color = Metadata_heart_number)) +
  geom_point(size = 2, alpha = 0.3) +
  scale_color_manual(name = "Heart Number",   
        values = c(
          "2" = "#00FF00",     
          "7" = "#0000FF",     
          "19" = "#FFA500",   
          "23" = "#FF0000",    
          "29" = "#800080",    
          "4" = "#FF00FF"      
        ),
        breaks = c("2", "7", "19", "23", "29", "4"),  
        labels = c("2", "7", "19", "23", "29", "4")
  ) +
  labs(title = "UMAP of Plate 4 Morphology Space Comparing Between Heart Numbers", x = "UMAP0", y = "UMAP1") +
  theme_bw() +
  theme(
    # Increase title size
    plot.title = element_text(size = 18),
    
    # Increase axis text size
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    
    # Increase legend size
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    
    # # Increase point size
    # plot.margin = margin(1,1,1,1, "cm")
  ) +
  facet_wrap(Metadata_treatment ~ Metadata_cell_type)

# saving image
output_file <- file.path(paste0(output_umap_file, "_heart_number.png"))
ggsave(output_file, umap_heart_num_figure, dpi = 500, height = 10, width = 14)

print(umap_heart_num_figure)

umap_cell_count_figure <- 
  ggplot(umap_cp_df, aes(x = UMAP0, y = UMAP1)) +
  geom_point(aes(color = Metadata_Cell_Count), alpha = 0.5) +
  theme_bw() +
  scale_color_continuous(
    name = "Cell Count", 
    low = "light pink",
    high = "red"
  ) +
  labs(title = "UMAP of Plate 4 Morphology Space Comparing\nBetween Cell Counts (Well-Level)", x = "UMAP0", y = "UMAP1") +
  theme(
    # Increase title size
    plot.title = element_text(size = 20),
    
    # Increase axis text size
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    
    # Increase legend size
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
  ) +
  coord_fixed(1.1)

# saving image
output_file <- file.path(paste0(output_umap_file, "_well_cell_count.png"))
ggsave(output_file, umap_cell_count_figure, dpi = 500, height = 10, width = 12)


print(umap_cell_count_figure)

# Set NA values as "None"
umap_cp_df$Metadata_treatment <- ifelse(is.na(umap_cp_df$Metadata_treatment), "None", umap_cp_df$Metadata_treatment)

umap_treatment_figure <- 
  ggplot(umap_cp_df, aes(x = UMAP0, y = UMAP1)) +
  geom_point(aes(color = Metadata_treatment), alpha = 0.5) +
  scale_color_manual(values = c("DMSO" = "#FF7F50", "None" = "#6B8EAD")) +
  theme_bw() +
  labs(title = "UMAP of Plate 4 Morphology Space Comparing\nBetween Treatments", x = "UMAP0", y = "UMAP1") +
  theme(
    # Increase title size
    plot.title = element_text(size = 20),
    
    # Increase axis text size
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    
    # Increase legend size
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
  )

# saving image
output_file <- file.path(paste0(output_umap_file, "_treatment.png"))
ggsave(output_file, umap_treatment_figure, dpi = 500, height = 10, width = 10)


print(umap_treatment_figure)

umap_cell_type_figure <- 
  ggplot(umap_cp_df, aes(x = UMAP0, y = UMAP1)) +
  geom_point(aes(color = Metadata_cell_type), alpha = 0.3) +
  scale_color_manual(values = c("Failing" = "magenta", "Healthy" = "darkgreen")) +
  theme_bw() +
  labs(title = "UMAP of Plate 4 Morphology Space Comparing\nBetween Cell Types", x = "UMAP0", y = "UMAP1") +
  theme(
    # Increase title size
    plot.title = element_text(size = 20),
    
    # Increase axis text size
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    
    # Increase legend size
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
  )

# saving image
output_file <- file.path(paste0(output_umap_file, "_cell_types.png"))
ggsave(output_file, umap_cell_type_figure, dpi = 500, height = 10, width = 10)


print(umap_cell_type_figure)

# Remove decimal from nuclei object number to avoid issues with merging
umap_cp_df <- umap_cp_df %>%
  mutate(Metadata_Nuclei_Number_Object_Number = gsub("\\.0", "", as.character(Metadata_Nuclei_Number_Object_Number)))

# Load in parquet file with nearest neighbors per single cell (not normalized)
neighbors_df <- arrow::read_parquet(
    "../../../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_annotated.parquet"
)

desired_columns <- c("Cells_Neighbors_NumberOfNeighbors_Adjacent", 
                     "Metadata_Well", 
                     "Metadata_Site", 
                     "Metadata_Nuclei_Number_Object_Number")

neighbors_df <- neighbors_df[, desired_columns]

neighbors_df <- neighbors_df %>%
  mutate(Metadata_Nuclei_Number_Object_Number = as.character(Metadata_Nuclei_Number_Object_Number))

# Assuming you want to merge based on the specified columns
merge_columns <- c("Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number")

# Merge number of neighbors based on cell to umap dataframe
merged_data <- merge(neighbors_df, umap_cp_df, by = merge_columns)

umap_cell_neighbors_figure <- 
  ggplot(merged_data, aes(x = UMAP0, y = UMAP1)) +
  geom_point(aes(color = Cells_Neighbors_NumberOfNeighbors_Adjacent), alpha = 0.5) +
  theme_bw() +
  scale_color_continuous(
    name = "Number of Cell Neighbors", 
    low = "light blue",
    high = "blue"
  ) +
  labs(title = "UMAP of Plate 4 Morphology Space Comparing\nBetween Number of Neighbors per Cell", x = "UMAP0", y = "UMAP1") +
  theme(
    # Increase title size
    plot.title = element_text(size = 20),
    
    # Increase axis text size
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    
    # Increase legend size
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
  ) +
  coord_fixed(1.1)

# saving image
output_file <- file.path(paste0(output_umap_file, "_number_neighbors_cells.png"))
ggsave(output_file, umap_cell_neighbors_figure, dpi = 500, height = 10, width = 12)


print(umap_cell_neighbors_figure)

