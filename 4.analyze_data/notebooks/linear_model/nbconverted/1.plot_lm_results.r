suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))


# path to tsv with LM coefficients
lm_results_dir <- file.path("./results/")
lm_file <- file.path(lm_results_dir, "localhost220512140003_KK22-05-198_linear_model_dose_count.tsv")

# save path for figure
lm_fig_dir <- file.path("./figures")
if (!dir.exists(lm_fig_dir)) {
    dir.create(lm_fig_dir, recursive = TRUE)
}
lm_fig <- file.path(lm_fig_dir, "linear_model_cp_features_heart3_dose_count.png")

# Load and process linear model data
lm_df <- readr::read_tsv(
    lm_file,
    col_types = readr::cols(.default = "d", Feature = "c")
)

print(dim(lm_df))
head(lm_df)


# Arrange by absolute value coefficient
# Split out components of feature name for visualization
lm_df <- lm_df %>%
    dplyr::arrange(desc(abs(dose_coef))) %>%
    tidyr::separate(
        Feature,
        into = c(
            "compartment",
            "feature_group",
            "measurement",
            "channel",
            "parameter1",
            "parameter2",
            "parameter3"
        ),
        sep = "_",
        remove = FALSE
    ) %>%
    dplyr::mutate(channel_cleaned = channel)

lm_df$channel_cleaned <- dplyr::recode(lm_df$channel_cleaned,
    "Hoechst" = "Nucleus",
    "ER" = "ER",
    "Actin" = "Actin",
    "Mitochondria" = "Mito",
    "PM" = "PM",
    .default = "other",
    .missing = "other"
)

print(dim(lm_df))
head(lm_df, 3)


# Drop rows with specific feature_group values
lm_df <- lm_df %>%
    dplyr::filter(!feature_group %in% c("Neighbors", "Location", "Parent", "Children", "Number"))

print(dim(lm_df))
head(lm_df, 3)


width <- 8
height <- 7
options(repr.plot.width = width, repr.plot.height = height)

# Plot the linear model coefficients
lm_fig_gg <- (
    ggplot(lm_df, aes(x = cell_count_coef, y = dose_coef))
    +
        geom_point(aes(size = r2_score, color = factor(feature_group), shape = channel_cleaned), alpha = 0.7)
        +
        geom_vline(xintercept = 0, linetype = "dashed", color = "red")
        +
        geom_density2d(color = "black", show.legend = FALSE, alpha = 0.7)
        +
        theme_bw()
        +
        theme(
            axis.title = element_text(size = 14), # Increase axis title font size
            axis.text = element_text(size = 12), # Increase axis tick font size
            legend.title = element_text(size = 12), # Increase legend title font size
            legend.text = element_text(size = 10) # Increase legend text font size
        )
        +
        guides(
            color = guide_legend(title = "Measurement", order = 1),
            size = guide_legend(title = "R2 score"),
            shape = guide_legend(title = "Channel\n(if applicable)", order = 2)
        )
        +
        ylab("drug_x dose contribution\n(LM beta coefficient)")
        +
        xlab("Cell count contribution\n(LM beta coefficient)")
)

# Output figure as PNG
ggsave(lm_fig, lm_fig_gg, dpi = 500, height = height, width = width)

# Output figure as PDF
lm_fig_pdf <- sub("\\.png$", ".pdf", lm_fig)
ggsave(lm_fig_pdf, lm_fig_gg, dpi = 500, height = height, width = width)

# Show figure
lm_fig_gg

