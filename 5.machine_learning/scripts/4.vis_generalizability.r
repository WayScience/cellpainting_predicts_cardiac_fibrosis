suppressPackageStartupMessages(suppressWarnings(library(ggplot2))) # plotting
suppressPackageStartupMessages(suppressWarnings(library(dplyr))) # data manipulation
suppressPackageStartupMessages(suppressWarnings(library(ggridges))) # ridge line plots
suppressPackageStartupMessages(suppressWarnings(library(RColorBrewer))) # color palettes

# load in the probabilities
combined_probabilities_path <- file.path(
    paste0(
        "./prob_data/combined_plate_3_predicted_proba.csv"
    )
)
# read in the data from the parquet file
combined_probabilities_df <- read.csv(
    combined_probabilities_path
)
dim(combined_probabilities_df)
head(combined_probabilities_df,2)

# Group by Metadata_cell_type
counts_df <- combined_probabilities_df %>%
  group_by(Metadata_cell_type) %>%
  summarise(count = n()) %>%
  ungroup()

head(counts_df)

# Group by Metadata_cell_type
counts_df <- combined_probabilities_df %>%
  group_by(Metadata_cell_type, predicted_label) %>%
  summarise(count = n()) %>%
  ungroup()

head(counts_df)

height <- 5
width <- 18
options(repr.plot.width = width, repr.plot.height = height)
# Adjust bandwidth parameter for density estimation to get more relative distributions for treatments with less cells
ridge_plot_healthy <- ggplot(combined_probabilities_df, aes(x = Healthy_probas, y = Metadata_cell_type, fill = Metadata_treatment)) +
  geom_density_ridges(aes(fill = Metadata_treatment), alpha = 0.7, scale = 3, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_manual(values = c(
    "drug_x" = brewer.pal(3, "Dark2")[1],
    "DMSO" = brewer.pal(3, "Dark2")[2],
    "TGFRi" = brewer.pal(3, "Dark2")[3]
  )) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ Metadata_treatment, scales = "free_y") +
  labs(y = "Heart Type") +
  theme_bw() +
  theme(legend.position = "none",
      axis.text = element_text(size = 12),
      axis.text.x = element_text(size = 12),
      axis.title = element_text(size = 14),
      strip.text = element_text(size = 14),
      strip.background = element_rect(
          colour = "black",
          fill = "#fdfff4"
      )
    )
  

# Save figure
ggsave("./figures/healthy_prob_ridge_plot_plate_3.png", ridge_plot_healthy, height = 8, width = 12, dpi = 500)

ridge_plot_healthy

bar_plot_predictions <- (
  ggplot(combined_probabilities_df, aes(x = Metadata_cell_type)) +
  geom_bar(aes(fill = predicted_label), position = "dodge") +
  labs(x = "Metadata_cell_type", y = "Count") +
  facet_grid(model_type~Metadata_treatment) +
  scale_fill_manual(values = c(
        "Healthy" = brewer.pal(3, "Dark2")[1],
        "Failing" = brewer.pal(3, "Dark2")[2]
    )) +
  theme_bw()
)
bar_plot_predictions

combined_probabilities_df_filtered <- combined_probabilities_df %>%
  filter(Healthy_probas > 0.9 | Failing_probas > 0.9)

bar_plot_predictions_filtered <- (
  ggplot(combined_probabilities_df_filtered, aes(x = Metadata_cell_type)) +
  geom_bar(aes(fill = predicted_label), position = "dodge") +
  labs(x = "Metadata_cell_type", y = "Count") +
  facet_grid(model_type~Metadata_treatment) +
  scale_fill_manual(values = c(
        "Healthy" = brewer.pal(3, "Dark2")[1],
        "Failing" = brewer.pal(3, "Dark2")[2]
    )) +
  theme_bw()
)

bar_plot_predictions_filtered

