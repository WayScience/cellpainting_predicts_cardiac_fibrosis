suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(platetools))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(RColorBrewer))

path_to_outlier_results <- file.path("./qc_results/qc_outliers.csv")
path_to_cp_qc_results <- file.path("./qc_results/localhost231120090001/Image.csv")

outlier_df <- read.csv(path_to_outlier_results)
qc_df <- read.csv(path_to_cp_qc_results)

# Add 'Metadata_Well' and 'Metadata_Site' columns using filename from 'FileName_OrigActin'
qc_df <- qc_df %>%
  mutate(Metadata_Well = gsub("_", "", str_extract(FileName_OrigActin, "_([^f]+)")),
         Metadata_Site = str_extract(FileName_OrigActin, "f.{2}"))
         
head(qc_df)

# Find the number of FOVs that are being removed due to being poor images
unique_combinations <- outlier_df[c('Metadata_Well', 'Metadata_Site')] %>% distinct()
cat("The number of image sets being removed is", nrow(unique_combinations), "out of 960 image sets\n")

# Remove the image sets from the main data frame to generate bar chart
result_df <- anti_join(qc_df, unique_combinations, by = c('Metadata_Well', 'Metadata_Site'))

# Group by 'Metadata_Well' and count the occurrences of 'Metadata_Site'
counts_per_well <- result_df %>% group_by(Metadata_Well) %>% summarise(count = n())

# Extract the first letter from 'Metadata_Well' to create a color palette
unique_starting_letters <- unique(substr(counts_per_well$Metadata_Well, 1, 1))

# Create a color palette based on unique starting letters
color_palette <- colorRampPalette(brewer.pal(8, "Set1"))(length(unique_starting_letters))

# Map each unique starting letter to a color in a named vector
color_dict <- setNames(color_palette, unique_starting_letters)

# Create a new column 'Color' in the data frame based on the starting letter
counts_per_well$Color <- color_dict[substr(counts_per_well$Metadata_Well, 1, 1)]

# Increase the figure size to extend the chart horizontally
options(repr.plot.width=14, repr.plot.height=6)

# Output path for bar chart
output_file <- "./qc_figures/localhost231120090001_fov_per_well_chart.png"

# Create a bar chart using ggplot2 with the 'Color' column for colors
fov_chart <- ggplot(counts_per_well, aes(x = Metadata_Well, y = count, fill = Color)) +
  geom_bar(stat = 'identity') +
  labs(x = 'Metadata_Well', y = 'Number of FOVs', title = 'Count of FOVs per Well for plate localhost231120090001') +
  scale_fill_identity() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save plot to qc_figures
ggsave(
        output_file,
        fov_chart,
        dpi = 500,
        height = 6,
        width = 10
    )

# Display the plot in the notebook
print(fov_chart)


# Output path for platemap
output_file <- "./qc_figures/localhost231120090001_platemap_fov_per_well.png"

fov_platemap <- platetools::raw_map(
    data = as.character(counts_per_well$count),
    well = counts_per_well$Metadata_Well,
    plate = 96,
    size = 8
    ) +

    ggtitle(paste("Platemap of FOV Count Per Well in plate localhost231120090001")) +
    theme(plot.title = element_text(size = 10, face = "bold")) +
    scale_fill_manual(
    name = "FOV Count",
    values = c("13" = "#adc8e6", "14" = "#4293f0", "15" = "#3b56f5", "16" = "darkblue")  # Assign colors manually
  )

    ggsave(
    output_file,
    fov_platemap,
    dpi = 500,
    height = 3.5,
    width = 6
    )

# Display the plot in the notebook
print(fov_platemap)
