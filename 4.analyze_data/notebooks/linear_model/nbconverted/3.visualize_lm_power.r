suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

plate <- "localhost220513100001_KK22-05-198_FactinAdjusted"  # Focus on plate 2

# Input files
lm_results_file <- file.path("results", paste0(plate, "_linear_model_cp_features.tsv"))
lm_power_file <- file.path("results", paste0(plate, "_power_analysis_cp_features_lm.tsv"))

# Output files
output_power_figure <- file.path("figures", paste0(plate, "_power_analysis_cp_lm.png"))

# Merge linear model results
lm_results_df <- readr::read_tsv(
    lm_results_file,
    col_types = readr::cols(.default="d", feature="c")
)
lm_power_df <- readr::read_tsv(
    lm_power_file,
    col_types = readr::cols(.default="d", feature="c")
)

# Merge for visualization
lm_data_df <- lm_results_df %>%
    dplyr::left_join(lm_power_df, by = "feature") %>%
    dplyr::arrange(estimated_sample_size) %>%
    dplyr::filter(estimated_sample_size < 100000) %>%  # Remove extreme outliers
    dplyr::filter(!is.na(power))

print(dim(lm_data_df))
head(lm_data_df)

# Load feature data (for calculating n)
file_suffix = "_sc_norm_fs_cellprofiler.csv.gz"
data_dir = file.path("..", "..", "..", "3.process-cfret-features", "data")
cp_file <- file.path(data_dir, paste0(plate, file_suffix))

cp_df <- readr::read_csv(
    cp_file,
    col_types = readr::cols(
        .default="d",
        Metadata_WellRow="c",
        Metadata_WellCol="c",
        Metadata_heart_number="c",
        Metadata_treatment="c",
        Metadata_dose="c",
        Metadata_Plate="c",
        Metadata_Well="c"
    )
)

n_samples <- dim(cp_df)[1]

print(dim(cp_df))
head(cp_df, 3)

power_gg <- (
    ggplot(lm_data_df, aes(y = estimated_sample_size, x = abs(treatment_dose_coef)))
    + geom_point(aes(color = abs(cell_count_coef)), size = 0.8)
    + theme_bw()
    + xlab("Treatment dose contribution\n(Absolute value LM beta coefficient)")
    + ylab("Sample size to acheive 80% power\nwith Bonferroni adjusted significance threshold")
    + ggtitle("Power analysis for CP features")
    + geom_hline(yintercept = n_samples, linetype = "dashed", color = "red")
    + scale_color_gradient(name = "Cell density\ncontribution", low = "black", high = "lightblue")
)

power_gg

power_zoom_gg <- (
    ggplot(
        lm_data_df %>% dplyr::filter(abs(treatment_dose_coef) > 0.05),
        aes(y = estimated_sample_size, x = abs(treatment_dose_coef))
    )
    + geom_point(
        aes(color = abs(cell_count_coef)),
        size = 0.8,
        show.legend = FALSE
    )
    + theme_bw()
    + xlab("Dose contribution (beta)")
    + ylab("Sufficient sample\nsize (80% power)")
    + ggtitle("Zoom in")
    + scale_color_gradient(name = "Cell density\ncontribution", low = "black", high = "lightblue")
)

power_zoom_gg

# Combine plots
combined_gg <- cowplot::ggdraw(
    power_gg
    + cowplot::draw_plot(
        power_zoom_gg,
        x = 0.08,
        y = 35000,
        width = .18,
        height = 35000
    )
)

ggsave(output_power_figure, combined_gg, width = 7, height = 6, dpi = 500)

combined_gg
