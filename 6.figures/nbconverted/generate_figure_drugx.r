invisible(lapply(
    c("dplyr", "ggplot2", "grid", "patchwork", "RColorBrewer", "ggExtra", "ggridges", "ggplotify"),
    function(pkg) suppressPackageStartupMessages(suppressWarnings(library(pkg, character.only = TRUE)))
))


figure_dir <- "./figures"
output_main_figure <- file.path(figure_dir, "main_figure.png")


workflow_path <- file.path("figures/CFReT_workflow.png")
workflow_img <- png::readPNG(workflow_path)

# Get the dimensions of the image
img_height <- nrow(workflow_img)
img_width <- ncol(workflow_img)

# Calculate the aspect ratio
aspect_ratio <- img_height / img_width

# Plot the workflow image from BioRender to a ggplot object
workflow <- ggplot() +
    annotation_custom(
        rasterGrob(workflow_img, interpolate = TRUE),
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
    ) +
    theme_void() +
    coord_fixed(ratio = aspect_ratio, clip = "off") +
    theme(plot.margin = margin(0, 0, 0, 0, "cm")) # Adjust margins as needed

workflow


montage_path <- file.path("figures/example_montage.png")
montage_img <- png::readPNG(montage_path)

# Get the dimensions of the image
img_height <- nrow(montage_img)
img_width <- ncol(montage_img)

# Calculate the aspect ratio
aspect_ratio <- img_height / img_width

# Plot the montage image from BioRender to a ggplot object
montage <- ggplot() +
    annotation_custom(
        rasterGrob(montage_img, interpolate = TRUE),
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
    ) +
    theme_void() +
    coord_fixed(ratio = aspect_ratio, clip = "off") +
    theme(plot.margin = margin(0, 0, 0, 0, "cm")) # Adjust margins as needed

montage


# Path to UMAP results
UMAP_results_dir <- file.path(
    "../4.analyze_data/notebooks/UMAP/results"
)

# Load data
UMAP_results_file <- file.path(UMAP_results_dir, "UMAP_localhost230405150001_DMSO_drugx.parquet")

UMAP_results_df <- arrow::read_parquet(UMAP_results_file)

dim(UMAP_results_df)
head(UMAP_results_df)


width <- 10
height <- 10
options(repr.plot.width = width, repr.plot.height = height)

# Create a new column combining cell type and treatment
UMAP_results_df <- UMAP_results_df %>%
    dplyr::mutate(Group = paste(Metadata_cell_type, Metadata_treatment, sep = " + "))

# Store the main UMAP plot as a ggplot object
umap_drugx_gg <- ggplot(UMAP_results_df, aes(x = UMAP0, y = UMAP1)) +
    geom_point(aes(color = Group), size = 0.9, alpha = 0.4) +
    geom_density_2d(aes(color = Group), alpha = 0.58, linewidth = 1.42) +
    # coord_fixed() +
    theme_bw() +
    theme(
        strip.background = element_rect(color = "black", fill = "#fdfff4"),
        legend.position = c(0.11, 0.91),
        legend.background = element_blank(),
        legend.key = element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(face = "bold"),
    ) +
    scale_color_manual(
        values = c("failing + drug_x" = "#E7298A", "failing + DMSO" = "#BA5A31", "healthy + DMSO" = "#8269dc", "healthy + drug_x" = "#E6AB02"),
    ) +
    guides(color = guide_legend(override.aes = list(size = 6))) +
    ylim(min(UMAP_results_df$UMAP1), max(UMAP_results_df$UMAP1)) +
    xlim(min(UMAP_results_df$UMAP0), max(UMAP_results_df$UMAP0)) +
    theme(
        legend.position.inside = c(0.85, 0.90),
        legend.background = element_blank(),
        legend.key = element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(size = 13, face = "bold"),
        panel.background = element_rect(fill = "white"),
        plot.background = element_blank(),
        axis.text = element_text(size = 22),
        axis.title = element_text(size = 24),
        axis.ticks = element_line(linewidth = 1.5)
    )

# Add density plots in the margins
merged_drugx_plot_with_margins <- ggMarginal(
    umap_drugx_gg,
    type = "density",
    margins = "both",
    groupFill = TRUE,
    linewidth = 5,
    colour = NA,
    size = 11,
)

# update to ggplot object
ggplot_drugx_plot_with_margins <- as.ggplot(merged_drugx_plot_with_margins)

# print plot
print(ggplot_drugx_plot_with_margins)


# Path to probability results
prob_results_dir <- file.path(
    "../5.machine_learning/3.assess_generalizability/prob_data"
)

# Load data
prob_results_file <- file.path(prob_results_dir, "proportion_healthy_plate3_grouped.csv")

prob_results_df <- readr::read_csv(prob_results_file)

# Remove TGFRi treatment and rename hearts to match manuscript
prob_results_df <- prob_results_df |>
    dplyr::filter(Metadata_treatment != "TGFRi") |>
    dplyr::mutate(
        Metadata_heart_number = dplyr::case_when(
            Metadata_heart_number == 9 ~ "F5",
            Metadata_heart_number == 11 ~ "NF2",
            TRUE ~ as.character(Metadata_heart_number)
        )
    )

dim(prob_results_df)
head(prob_results_df)


prob_results_df <- prob_results_df |>
    dplyr::mutate(bar_group = dplyr::case_when(
        Metadata_heart_number == "F5" & Metadata_treatment == "drug_x" ~ "failing + drug_x",
        Metadata_heart_number == "F5" & Metadata_treatment == "DMSO" ~ "failing + DMSO",
        Metadata_heart_number == "NF2" & Metadata_treatment == "DMSO" ~ "healthy + DMSO",
        Metadata_heart_number == "NF2" & Metadata_treatment == "drug_x" ~ "healthy + drug_x"
    ))

width <- 8
height <- 8
options(repr.plot.width = width, repr.plot.height = height)

# Create a bar plot
bar_plot <- ggplot(prob_results_df, aes(x = Metadata_heart_number, y = proportion_healthy, fill = bar_group)) +
    geom_bar(stat = "identity", position = position_dodge(), color = "black") +
    theme_bw() +
    labs(
        x = "Heart number",
        y = "Proportion of single cells\npredicted as from a healthy heart"
    ) +
    theme(
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        legend.position = "none"
    ) +
    scale_fill_manual(
        values = c(
            "failing + drug_x" = "#E7298A",
            "failing + DMSO" = "#BA5A31",
            "healthy + DMSO" = "#8269dc",
            "healthy + drug_x" = "#E6AB02"
        )
    )

# Print the plot
bar_plot


# Path to probability results for pilot data (plates 1 and 2)
pilot_results_dir <- file.path(
  "../5.machine_learning/3.assess_generalizability/prob_data"
)

# Load data
pilot_results_file <- file.path(pilot_results_dir, "combined_plates_1_2_predicted_proba.csv")

pilot_results_df <- readr::read_csv(pilot_results_file)

# Update the heart numbers to match manuscript
pilot_results_df <- pilot_results_df %>%
  mutate(Metadata_heart_number = recode(Metadata_heart_number, `3` = "F2", `8` = "F4", `9` = "F5"))

# Add "uM" to all values in the Metadata_dose column
pilot_results_df$Metadata_dose <- paste0(as.character(pilot_results_df$Metadata_dose), "\nuM")

# Specify the levels in the desired order
dose_levels <- c("0\nuM", "0.005\nuM", "0.014\nuM", "0.04\nuM", "0.12\nuM", "0.37\nuM", "1.11\nuM", "3.33\nuM", "5\nuM", "10\nuM")

# Convert Metadata_dose to factor with specified levels
pilot_results_df$Metadata_dose <- factor(pilot_results_df$Metadata_dose, levels = dose_levels)

dim(pilot_results_df)
head(pilot_results_df)


height <- 20
width <- 5
options(repr.plot.width = width, repr.plot.height = height)

# Define colors for the doses
zero_dose_color <- "#1B9E77" # Unique color for 0 uM
all_other_doses_color <- "#4C6FA3" # Same color for all other doses
dose_colors <- c(
    "0\nuM" = zero_dose_color,
    "0.005\nuM" = all_other_doses_color,
    "0.014\nuM" = all_other_doses_color,
    "0.04\nuM" = all_other_doses_color,
    "0.12\nuM" = all_other_doses_color,
    "0.37\nuM" = all_other_doses_color,
    "1.11\nuM" = all_other_doses_color,
    "3.33\nuM" = all_other_doses_color,
    "5\nuM" = all_other_doses_color,
    "10\nuM" = all_other_doses_color
)

# Basic plot with custom colors
ridge_plot_healthy <- ggplot(pilot_results_df, aes(
    x = Healthy_probas,
    y = Metadata_heart_number,
    fill = Metadata_dose
)) +
    geom_density_ridges(alpha = 0.7, scale = 3, rel_min_height = 0.01, bandwidth = 0.1) +
    scale_fill_manual(values = dose_colors) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
    scale_x_continuous(breaks = seq(0, 1, 0.5)) +
    facet_grid(Metadata_dose ~ model_type, scales = "free_y") +
    labs(
        y = "Heart number",
        x = "Healthy heart\nprobability",
    ) +
    theme_bw() +
    theme(
        legend.position = "none",
        axis.text = element_text(size = 20),
        axis.text.x = element_text(size = 20),
        axis.title = element_text(size = 24),
        strip.text = element_text(size = 24),
        strip.background = element_rect(
            colour = "black",
            fill = "#fdfff4"
        )
    )

# Show plot
ridge_plot_healthy


width <- 12
height <- 12
options(repr.plot.width = width, repr.plot.height = height)

panels_a_b_gg <- (
    workflow +
        montage
) + plot_layout(heights = c(1, 1))


width <- 12
height <- 12
options(repr.plot.width = width, repr.plot.height = height)

panels_c_d_gg <- (
    ggplot_drugx_plot_with_margins +
        free(bar_plot)
) + plot_layout(heights = c(1.2, 1))


width <- 26
height <- 16
options(repr.plot.width = width, repr.plot.height = height)

panels_a_b_c_d_gg <- (
    panels_a_b_gg |
        free(panels_c_d_gg)
) + plot_layout(widths = c(1.5, 1))


width <- 32
height <- 16
options(repr.plot.width = width, repr.plot.height = height)

main_figure <- (
    panels_a_b_c_d_gg |
        free(ridge_plot_healthy)
) +
    plot_layout(widths = c(2.15, 1.25, 1)) +
    plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 40))

# Save figure
ggsave(output_main_figure, main_figure, height = height, width = width, dpi = 500)

main_figure

