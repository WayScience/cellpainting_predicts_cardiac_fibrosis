suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(RColorBrewer))

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

# Load the necessary data file
prob_df <- read.csv("prob_data/combined_plate_3_predicted_proba.csv")

# Filter for rows where model_type is "final"
final_prob_df <- subset(prob_df, model_type == "final")

# Reset row names to default sequence
row.names(final_prob_df) <- NULL

# Display the filtered dataframe
dim(final_prob_df)
head(final_prob_df)

# Check columns in final_prob_df but not in umap_cp_df
setdiff(colnames(final_prob_df), colnames(umap_cp_df))

# Check columns in umap_cp_df but not in final_prob_df
setdiff(colnames(umap_cp_df), colnames(final_prob_df))


# Ensure only the specific metadata columns are used for the join
metadata_cols <- c("Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number")

# Ensure these Metadata columns are character type, lowercase, and trimmed of whitespace in both dataframes
final_prob_df <- final_prob_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

umap_cp_df <- umap_cp_df %>%
  mutate(across(metadata_cols, ~ str_trim(as.character(.), side = "both")))

# Perform the join using the selected "Metadata" columns
combined_df <- umap_cp_df %>%
  left_join(final_prob_df, by = metadata_cols)

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
output_file <- paste0(output_umap_file, "_healthy_proba.png")

# Create the UMAP scatterplot
umap <- ggplot(combined_df, aes(x = UMAP0, y = UMAP1, color = Healthy_probas)) +
  geom_point(alpha = 0.3) +  # Set alpha for transparency (lower value means more transparent)
  scale_color_gradientn(
    colors = brewer.pal(9, "RdYlBu"),  # You can try other palettes like "Blues", "Greens", etc.
    limits = c(0, 1)  # Set the legend limits to be 0 to 1.0
  ) +  # Color gradient for continuous data
  labs(x = "UMAP0", y = "UMAP1", color = "Healthy\nProbability\n") +
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
