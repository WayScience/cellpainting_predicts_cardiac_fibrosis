suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))

# Focusing on plate 2
plate <- "localhost220513100001_KK22-05-198_FactinAdjusted"

input_dir <- "results"
lm_file <- file.path(input_dir, paste0(plate, "_linear_model_cp_features.tsv"))
lm_output_file <- file.path(input_dir, paste0(plate, "_linear_model_cp_features_cleaned_sorted.tsv"))

output_fig_dir <- "figures"
lm_fig <- file.path(output_fig_dir, paste0(plate, "_linear_model_cp_features.png"))
lm_facet_fig <- file.path(output_fig_dir, paste0(plate, "_linear_model_cp_features_bygroup.png"))

# Load and process linear model data
lm_df <- readr::read_tsv(
    lm_file, col_types = readr::cols(.default = "d", feature = "c")
)

# Arrange by absolute value coefficient
# Split out components of feature name for visualization
lm_df <- lm_df %>%
    dplyr::arrange(desc(abs(treatment_dose_coef))) %>%
    tidyr::separate(
        feature,
        into = c(
            "compartment",
            "feature_group",
            "measurement",
            "channel", 
            "parameter1", 
            "parameter2"
        ),
        sep = "_",
        remove = FALSE
    ) %>%
    dplyr::mutate(channel_cleaned = channel) %>%
    dplyr::arrange(desc(abs(treatment_dose_coef)))

lm_df$channel_cleaned <- dplyr::recode(lm_df$channel_cleaned,
    "Hoechst" = "Nucleus",
    "ER" = "ER",
    "Actin" = "Actin",
    "Mitochondria" = "Mito",
    "Golgi" = "Golgi",
    .default = "other",
    .missing = "other"
)

# Output cleaned, parsed, and sorted file
readr::write_tsv(lm_df, lm_output_file)

print(dim(lm_df))
head(lm_df, 3)

lm_fig_gg <- (
    ggplot(lm_df, aes(x = cell_count_coef, y = treatment_dose_coef))
    + geom_point(aes(size = r2_score, color = channel_cleaned), alpha = 0.7)
    + geom_vline(xintercept = 0, linetype = "dashed", color = "red")
    + geom_density2d(color="black", show.legend = FALSE)
    + theme_bw()
    + guides(
        color = guide_legend(title = "Channel\n(if applicable)", order = 1),
        size = guide_legend(title = "R2 score")
    )
    + ylab("Treatment dose contribution (LM beta coefficient)")
    + xlab("Cell count contribution (LM beta coefficient)")
    + ggtitle("How CellProfiler features contribute\nto treatment and cell density")
)

# Save figure
ggsave(lm_fig, lm_fig_gg, height = 6, width = 6, dpi = 500)

lm_fig_gg

lm_facet_fig_gg <- (
    ggplot(lm_df, aes(x = cell_count_coef, y = treatment_dose_coef))
    + geom_point(aes(size = r2_score, color = feature_group), alpha = 0.7)
    + facet_wrap("~channel_cleaned")
    + geom_vline(xintercept = 0, linetype = "dashed", color = "red")
    + theme_bw()
    + guides(
        color = guide_legend(title = "Feature group\n(if applicable)", order = 1),
        size = guide_legend(title = "R2 score")
    )
    + ylab("Treatment dose contribution (LM beta coefficient)")
    + xlab("Cell count contribution (LM beta coefficient)")
    + ggtitle("How CellProfiler features (by group) contribute to treatment and cell density")
    + scale_color_brewer(palette="Dark2")
    + theme(
        axis.text = element_text(size = 7),
        axis.title = element_text(size = 10),
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 10),
        strip.text = element_text(size = 8),
        strip.background = element_rect(
            colour = "black",
            fill = "#fdfff4"
        )
    )
)

# Save figure
ggsave(lm_facet_fig, lm_facet_fig_gg, height = 6, width = 8, dpi = 500)

lm_facet_fig_gg
