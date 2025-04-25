suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(ggExtra))
suppressPackageStartupMessages(library(gifski))


pilot_data_umap_df <- read_parquet("./results/UMAP_plates_1_2_combined_KK22-05-198.parquet")

# Modify the Metadata_dose column to append 'uM' and preserve the order
pilot_data_umap_df$Metadata_dose <- paste0(pilot_data_umap_df$Metadata_dose, " uM")

# Remove " uM", sort the numeric values, and then append " uM" again
sorted_doses <- sort(as.numeric(gsub(" uM", "", pilot_data_umap_df$Metadata_dose)))

# Ensure no duplicates and restore " uM"
unique_sorted_doses <- unique(sorted_doses)
pilot_data_umap_df$Metadata_dose <- factor(
    pilot_data_umap_df$Metadata_dose,
    levels = paste0(unique_sorted_doses, " uM")
)

# Group by Metadata_Well and count cells
cell_count_df <- pilot_data_umap_df %>%
    dplyr::group_by(Metadata_Well) %>%
    dplyr::count() %>%
    dplyr::rename(Metadata_Cell_Count = n)

# Merge the cell count data with the original dataframe
pilot_data_umap_df <- pilot_data_umap_df %>%
    dplyr::left_join(cell_count_df, by = "Metadata_Well")

dim(pilot_data_umap_df)
head(pilot_data_umap_df)


# Create a folder to save the images
dir.create("umap_pilot_plates_frames", showWarnings = FALSE)

# Get all unique doses in order
doses <- pilot_data_umap_df$Metadata_dose %>%
    unique() %>%
    sort(by = ~ as.numeric(sub(" uM", "", .)))

# Compute axis and color limits across all data
x_limits <- range(pilot_data_umap_df$UMAP0, na.rm = TRUE)
y_limits <- range(pilot_data_umap_df$UMAP1, na.rm = TRUE)
color_limits <- range(pilot_data_umap_df$Metadata_Cell_Count, na.rm = TRUE)

# Loop through doses and save each plot
for (i in seq_along(doses)) {
    p <- ggplot(
        pilot_data_umap_df %>% filter(Metadata_dose == doses[i]),
        aes(x = UMAP0, y = UMAP1, color = Metadata_Cell_Count)
    ) +
        geom_point(alpha = 0.6) +
        ggtitle(paste("Dose:", doses[i])) +
        theme_bw() +
        xlim(x_limits) +
        ylim(y_limits) +
        scale_color_continuous(low = "lightblue", high = "darkblue", limits = color_limits) +
        labs(color = "Well-level\ncell counts")

    ggsave(filename = sprintf("umap_pilot_plates_frames/frame_%02d.png", i), plot = p, width = 6, height = 5, bg = "white")
}

# Make the GIF
gifski(
    png_files = list.files("umap_pilot_plates_frames", full.names = TRUE, pattern = "*.png"),
    gif_file = "umap_pilot_plates_frames/umap_pilot_plates_by_dose.gif",
    width = 600, height = 500, delay = 0.35
)
