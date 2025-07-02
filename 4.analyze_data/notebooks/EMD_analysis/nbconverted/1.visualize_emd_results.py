#!/usr/bin/env python
# coding: utf-8

# # Visualize EMD results for each comparison
# 
# We will visualize the signed EMD results for each comparison as violin and stacked bar plots.

# In[1]:


suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(arrow))


# ## Set output directory and constants for figures

# In[2]:


output_dir <- "./emd_figures"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Set thresholds for EMD values that represent "no change"
emd_positive_threshold <- 0.1
emd_negative_threshold <- -0.1


# ## Visualize failing DMSO cells (reference) compared to failing drug_x cells

# In[3]:


failing_vs_failing_drug_x_emd_df <- read_parquet("emd_results/failing_vs_failing_drug_x_emd.parquet")

dim(failing_vs_failing_drug_x_emd_df)
head(failing_vs_failing_drug_x_emd_df)


# In[4]:


height <- 10
width <- 20
options(repr.plot.width = width, repr.plot.height = height)

failing_vs_failing_drug_x_plot <- ggplot(failing_vs_failing_drug_x_emd_df, aes(x = feature_group, y = signed_emd)) +
    geom_violin(trim = FALSE, fill = "grey60", color = NA, alpha = 0.7) +
    geom_jitter(aes(color = organelle), width = 0.2, alpha = 0.4, size = 2.5) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.7, alpha = 0.5) +
    geom_hline(yintercept = emd_positive_threshold, linetype = "dashed", color = "#D62728", size = 0.7, alpha = 0.5) +
    geom_hline(yintercept = emd_negative_threshold, linetype = "dashed", color = "#D62728", size = 0.7, alpha = 0.5) +
    facet_wrap(~ compartment) +
    theme_bw() +
    labs(
        title = "Failing DMSO vs failing drug_x EMD",
        x = "Feature Group",
        y = "Signed EMD"
    ) +
    theme(
        legend.position = "bottom",
        plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16),
        axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 18),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18)
    )

# Save the plot
ggsave(
    filename = file.path(output_dir, "failing_vs_failing_drug_x_emd_violin_plot.png"),
    plot = failing_vs_failing_drug_x_plot,
    width = width,
    height = height,
    dpi = 500
)

failing_vs_failing_drug_x_plot


# In[5]:


emd_direction_fraction <- failing_vs_failing_drug_x_emd_df %>%
  mutate(direction = case_when(
    signed_emd > emd_positive_threshold ~ "increased",
    signed_emd < emd_negative_threshold ~ "decreased",
    TRUE ~ "no change"
  )) %>%
  group_by(feature_group, compartment, direction) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(feature_group, compartment) %>%
  mutate(fraction = count / sum(count)) %>%
  ungroup()


head(emd_direction_fraction)


# In[6]:


height <- 8
width <- 15
options(repr.plot.width = width, repr.plot.height = height)

emd_direction_fraction_plot <- ggplot(emd_direction_fraction, aes(x = fraction, y = feature_group, fill = direction)) +
    geom_bar(stat = "identity", position = "stack") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(
        title = "Failing DMSO vs failing drug_x EMD direction fraction",
        x = "Proportion of features",
        y = "Feature group",
        fill = "Direction"
    ) +
    facet_wrap(~ compartment, scales = "free_y") +
    theme_bw(base_size = 18) +
    scale_fill_manual(
    values = c(
        "increased" = "#D62728",  # red
        "decreased" = "#1F77B4",  # blue
        "no change" = "grey60"
        )
    ) +
    theme(
        plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 14)
    )

# Save the plot
ggsave(
    filename = file.path(output_dir, "failing_vs_failing_drug_x_emd_fraction_bar_plot.png"),
    plot = emd_direction_fraction_plot,
    width = width,
    height = height,
    dpi = 500
)

emd_direction_fraction_plot


# ## Visualize healthy DMSO cells (reference) compared to failing drug_x cells

# In[7]:


healthy_vs_failing_drug_x_emd_df <- read_parquet("emd_results/healthy_vs_failing_drug_x_emd.parquet")

dim(healthy_vs_failing_drug_x_emd_df)
head(healthy_vs_failing_drug_x_emd_df)


# In[8]:


height <- 10
width <- 20
options(repr.plot.width = width, repr.plot.height = height)

healthy_vs_failing_drug_x_plot <- ggplot(healthy_vs_failing_drug_x_emd_df, aes(x = feature_group, y = signed_emd)) +
    geom_violin(trim = FALSE, fill = "grey60", color = NA, alpha = 0.7) +
    geom_jitter(aes(color = organelle), width = 0.2, alpha = 0.4, size = 2.5) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.7, alpha = 0.5) +
    geom_hline(yintercept = emd_positive_threshold, linetype = "dashed", color = "#D62728", size = 0.7, alpha = 0.5) +
    geom_hline(yintercept = emd_negative_threshold, linetype = "dashed", color = "#D62728", size = 0.7, alpha = 0.5) +
    facet_wrap(~ compartment) +
    theme_bw() +
    labs(
        title = "Healthy DMSO vs failing drug_x EMD",
        x = "Feature Group",
        y = "Signed EMD"
    ) +
    theme(
        legend.position = "bottom",
        plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16),
        axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 18),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18)
    )

# Save the plot
ggsave(
    filename = file.path(output_dir, "healthy_vs_failing_drug_x_emd_violin_plot.png"),
    plot = healthy_vs_failing_drug_x_plot,
    width = width,
    height = height,
    dpi = 500
)

healthy_vs_failing_drug_x_plot


# In[9]:


hvf_emd_direction_fraction <- healthy_vs_failing_drug_x_emd_df %>%
  mutate(direction = case_when(
    signed_emd > emd_positive_threshold ~ "increased",
    signed_emd < emd_negative_threshold ~ "decreased",
    TRUE ~ "no change"
  )) %>%
  group_by(feature_group, compartment, direction) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(feature_group, compartment) %>%
  mutate(fraction = count / sum(count)) %>%
  ungroup()


head(hvf_emd_direction_fraction)


# In[10]:


height <- 8
width <- 15
options(repr.plot.width = width, repr.plot.height = height)

hvf_emd_direction_fraction_plot <- ggplot(hvf_emd_direction_fraction, aes(x = fraction, y = feature_group, fill = direction)) +
    geom_bar(stat = "identity", position = "stack") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(
        title = "Healthy DMSO vs failing drug_x EMD direction fraction",
        x = "Fraction of measurements",
        y = "Feature group",
        fill = "Direction"
    ) +
    facet_wrap(~ compartment, scales = "free_y") +
    theme_bw(base_size = 18) +
    scale_fill_manual(
    values = c(
        "increased" = "#D62728",  # red
        "decreased" = "#1F77B4",  # blue
        "no change" = "grey60"
        )
    ) +
    theme(
        plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 14)
    )

# Save the plot
ggsave(
    filename = file.path(output_dir, "healthy_vs_failing_drug_x_emd_fraction_bar_plot.png"),
    plot = hvf_emd_direction_fraction_plot,
    width = width,
    height = height,
    dpi = 500
)

hvf_emd_direction_fraction_plot


# ## Visualize failing DMSO cells (reference) compared to healthy DMSO cells

# In[11]:


failing_vs_healthy_DMSO_emd_df <- read_parquet("emd_results/failing_vs_healthy_DMSO_emd.parquet")

dim(failing_vs_healthy_DMSO_emd_df)
head(failing_vs_healthy_DMSO_emd_df)


# In[12]:


height <- 10
width <- 20
options(repr.plot.width = width, repr.plot.height = height)

failing_vs_healthy_DMSO_plot <- ggplot(failing_vs_healthy_DMSO_emd_df, aes(x = feature_group, y = signed_emd)) +
    geom_violin(trim = FALSE, fill = "grey60", color = NA, alpha = 0.7) +
    geom_jitter(aes(color = organelle), width = 0.2, alpha = 0.4, size = 2.5) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.7, alpha = 0.5) +
    geom_hline(yintercept = emd_positive_threshold, linetype = "dashed", color = "#D62728", size = 0.7, alpha = 0.5) +
    geom_hline(yintercept = emd_negative_threshold, linetype = "dashed", color = "#D62728", size = 0.7, alpha = 0.5) +
    facet_wrap(~ compartment) +
    theme_bw() +
    labs(
        title = "Failing DMSO vs healthy DMSO EMD",
        x = "Feature Group",
        y = "Signed EMD"
    ) +
    theme(
        legend.position = "bottom",
        plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16),
        axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 18),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18)
    )

# Save the plot
ggsave(
    filename = file.path(output_dir, "failing_vs_healthy_DMSO_emd_violin_plot.png"),
    plot = failing_vs_healthy_DMSO_plot,
    width = width,
    height = height,
    dpi = 500
)

failing_vs_healthy_DMSO_plot


# In[13]:


fvh_emd_direction_fraction <- failing_vs_healthy_DMSO_emd_df %>%
  mutate(direction = case_when(
    signed_emd > emd_positive_threshold ~ "increased",
    signed_emd < emd_negative_threshold ~ "decreased",
    TRUE ~ "no change"
  )) %>%
  group_by(feature_group, compartment, direction) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(feature_group, compartment) %>%
  mutate(fraction = count / sum(count)) %>%
  ungroup()


head(fvh_emd_direction_fraction)


# In[14]:


height <- 8
width <- 15
options(repr.plot.width = width, repr.plot.height = height)

fvh_emd_direction_fraction_plot <- ggplot(fvh_emd_direction_fraction, aes(x = fraction, y = feature_group, fill = direction)) +
    geom_bar(stat = "identity", position = "stack") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(
        title = "Failing DMSO vs healthy DMSO EMD direction fraction",
        x = "Fraction of measurements",
        y = "Feature group",
        fill = "Direction"
    ) +
    facet_wrap(~ compartment, scales = "free_y") +
    theme_bw(base_size = 18) +
    scale_fill_manual(
    values = c(
        "increased" = "#D62728",  # red
        "decreased" = "#1F77B4",  # blue
        "no change" = "grey60"
        )
    ) +
    theme(
        plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 14)
    )

# Save the plot
ggsave(
    filename = file.path(output_dir, "failing_vs_healthy_DMSO_emd_fraction_bar_plot.png"),
    plot = fvh_emd_direction_fraction_plot,
    width = width,
    height = height,
    dpi = 500
)

fvh_emd_direction_fraction_plot

