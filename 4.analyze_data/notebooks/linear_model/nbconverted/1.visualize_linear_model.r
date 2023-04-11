suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))

# Focusing on plate 2
plate <- "localhost220513100001_KK22-05-198_FactinAdjusted"

input_dir <- "results"
lm_file <- file.path(input_dir, paste0(plate, "_linear_model_cp_features.tsv"))

output_fig_dir <- "figures"
lm_fig <- file.path(output_fig_dir, paste0(plate, "_linear_model_cp_features.png"))

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
    dplyr::mutate(channel_cleaned = channel)

lm_df$channel_cleaned <- dplyr::recode(lm_df$channel_cleaned,
    "Hoechst" = "Nucleus",
    "ER" = "ER",
    "Actin" = "Actin",
    "Mitochondria" = "Mito",
    "Golgi" = "Golgi",
    .default = "other",
    .missing = "other"
)

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

lm_fig_gg
