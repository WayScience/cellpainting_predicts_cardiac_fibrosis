#!/usr/bin/env python
# coding: utf-8

# # Visualize ML model coefficients per class

# In[1]:


suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))


# In[2]:


# Dir with coef files
input_data_dir <- "./data"

# Set output figure dir
output_fig_dir <- "./figures"

# Path to positive coef file (features related to predicting Healthy class)
healthy_coef_file <- file.path(input_data_dir, "positive_coeffs.csv")

# Path to negative coef file (features related to predicting Failing class)
failing_coef_file <- file.path(input_data_dir, "negative_coeffs.csv")

# Paths for each figure output
healthy_coef_fig <- file.path(output_fig_dir, "logistic_regression_final_healthy_coefficients_plot.png")
failing_coef_fig <- file.path(output_fig_dir, "logistic_regression_final_failing_coefficients_plot.png")


# ## Load in and clean data frames to use in plotting

# In[3]:


# Load and clean healthy class data frame
healthy_coef_df <- readr::read_csv(
    healthy_coef_file, col_types = readr::cols(.default = "d", Feature = "c")
)

# Split out components of feature name for visualization
healthy_coef_df <- healthy_coef_df %>%
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

healthy_coef_df$channel_cleaned <- dplyr::recode(healthy_coef_df$channel_cleaned,
    "Hoechst" = "Nucleus",
    "ER" = "ER",
    "Actin" = "Actin",
    "Mitochondria" = "Mito",
    "PM" = "PM",
    .default = "other",
    .missing = "other"
)

print(dim(healthy_coef_df))
head(healthy_coef_df, 3)


# In[4]:


# Load and clean failing class data frame
failing_coef_df <- readr::read_csv(
    failing_coef_file, col_types = readr::cols(.default = "d", Feature = "c")
)

# Split out components of feature name for visualization
failing_coef_df <- failing_coef_df %>%
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

failing_coef_df$channel_cleaned <- dplyr::recode(failing_coef_df$channel_cleaned,
    "Hoechst" = "Nucleus",
    "ER" = "ER",
    "Actin" = "Actin",
    "Mitochondria" = "Mito",
    "PM" = "PM",
    .default = "other",
    .missing = "other"
)

print(dim(failing_coef_df))
head(failing_coef_df, 3)


# ## Plot coefficients for each cell type

# In[5]:


healthy_coef_gg <- (
    ggplot(healthy_coef_df, aes(x = channel_cleaned, y = feature_group))
    + geom_point(aes(fill = Healthy_Coeffs), pch = 22, size = 5)
    + facet_wrap("~compartment", ncol = 1)
    + theme_bw()
    + scale_fill_gradient(
        name="Top healthy cell type\nML model\ncoefficient",
        low = "darkblue",
        high = "yellow"
    )
    + xlab("Channel")
    + ylab("Feature")
    + theme(
        axis.text = element_text(size = 7),
        axis.text.x = element_text(angle = 90, size = 7),
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
ggsave(healthy_coef_fig, healthy_coef_gg, height = 8, width = 8, dpi = 500)

healthy_coef_gg


# In[6]:


failing_coef_gg <- (
    ggplot(failing_coef_df, aes(x = channel_cleaned, y = feature_group))
    + geom_point(aes(fill = abs(Failing_Coeffs)), pch = 22, size = 5)
    + facet_wrap("~compartment", ncol = 1)
    + theme_bw()
    + scale_fill_gradient(
        name="Top healthy cell type\nML model\ncoefficient",
        low = "darkblue",
        high = "yellow"
    )
    + xlab("Channel")
    + ylab("Feature")
    + theme(
        axis.text = element_text(size = 7),
        axis.text.x = element_text(angle = 90, size = 7),
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
ggsave(failing_coef_fig, failing_coef_gg, height = 8, width = 8, dpi = 500)

failing_coef_gg

