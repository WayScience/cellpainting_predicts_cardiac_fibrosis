#!/usr/bin/env python
# coding: utf-8

# # Calculate earth movers distance per feature between populations
# 
# Inspired by the work done in the [SPACe paper](https://www.nature.com/articles/s41467-024-54264-4#Sec10), we are calculating the earth mover's distance with a sign based on if the median value of a feature is higher or lower than the reference population. The populations we are comparing are:
# 
# 1. Failing + DMSO (reference) to Failing + drug_x
# 2. Healthy + DMSO (reference) to Failing + drug_x
# 3. Failing + DMSO (reference) to Healthy + DMSO
# 
# We are using SciPy's implementation of calculating earth mover's distance also known as Wasserstein distance.

# In[1]:


import pathlib
import pandas as pd

from scipy.stats import wasserstein_distance
import numpy as np

import sys

sys.path.append("../../../utils")

from emd_utils import compute_signed_emd_per_feature, compute_null_emd_range


# ## Set output directory for calculated EMD per comparison

# In[2]:


output_dir = pathlib.Path("./emd_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Output file for EMD thresholds
output_emd_thresholds = pathlib.Path(f"{output_dir}/emd_thresholds.csv")
# Define the columns you want
columns = ["Reference", "Comparison", "EMD_threshold_lower", "EMD_threshold_upper"]
# Create an empty DataFrame with just the header
pd.DataFrame(columns=columns).to_csv(output_emd_thresholds, index=False)


# ## Load in data with drug_x and controls

# In[3]:


# Directory containing the data files
data_dir = pathlib.Path("../../../3.process_cfret_features/data/single_cell_profiles")

# Load in "plate 3" data that contains the control and drug_x conditions
data_df = pd.read_parquet(
    pathlib.Path(data_dir, "localhost230405150001_sc_normalized.parquet")
)

# Drop columns that don't start with Metadata_ and contain 'Location', 'Parent', or 'Child'
cols_to_drop = [
    col
    for col in data_df.columns
    if not col.startswith("Metadata_")
    and (
        "Location" in col
        or "Parent" in col
        or "Child" in col
        or "Number" in col
        or "Neighbors" in col
    )  # Drop neighbors due to low feature numbers
]
data_df = data_df.drop(columns=cols_to_drop)

# Print dataframe
print(data_df.shape)
data_df.head()


# ## Split data into the three populations to compare

# In[4]:


# Split the data into each population/condition
healthy_DMSO_df = data_df[
    (data_df["Metadata_cell_type"] == "healthy")
    & (data_df["Metadata_treatment"] == "DMSO")
]
failing_drug_x_df = data_df[
    (data_df["Metadata_cell_type"] == "failing")
    & (data_df["Metadata_treatment"] == "drug_x")
]
failing_DMSO_df = data_df[
    (data_df["Metadata_cell_type"] == "failing")
    & (data_df["Metadata_treatment"] == "DMSO")
]

# Print the shapes of the dataframes
print("Healthy DMSO shape:", healthy_DMSO_df.shape)
print("Failing Drug X shape:", failing_drug_x_df.shape)
print("Failing DMSO shape:", failing_DMSO_df.shape)


# ## Compute EMD comparing failing drug_x cells to the healthy DMSO cells (reference)

# In[5]:


# Reference query to split the shuffled data
reference_query = '(Metadata_cell_type == "healthy") & (Metadata_treatment == "DMSO")'

# Compute the baseline null range for EMD (5th and 95th percentiles)
hvfd_threshold_lower, hvfd_threshold_upper = compute_null_emd_range(
    reference_df=healthy_DMSO_df,
    comparison_df=failing_drug_x_df,
    reference_query=reference_query,
    num_permutations=20,
)

# Create one-row dataframe to save threshold range to CSV
hvfd_row = pd.DataFrame(
    [
        {
            "Reference": "healthy_DMSO",
            "Comparison": "failing_drug_x",
            "EMD_threshold_lower": hvfd_threshold_lower,
            "EMD_threshold_upper": hvfd_threshold_upper,
        }
    ]
)

# Append to existing CSV file (no header, append mode)
print("Writing to:", output_emd_thresholds.resolve())
hvfd_row.to_csv(output_emd_thresholds, mode="a", index=False, header=False)

# Print the threshold range
print(f"EMD threshold range: [{hvfd_threshold_lower:.4f}, {hvfd_threshold_upper:.4f}]")


# In[6]:


# Compute the signed EMD for each feature
healthy_vs_failing_drug_x_emd = compute_signed_emd_per_feature(
    reference_df=healthy_DMSO_df, comparison_df=failing_drug_x_df
)

# Print the results
print(healthy_vs_failing_drug_x_emd.shape)
healthy_vs_failing_drug_x_emd.sort_values("signed_emd", ascending=False).head()


# ### Split feature column into parts

# In[7]:


# Split the 'feature' column into new columns as requested (not as a function)
split_df = healthy_vs_failing_drug_x_emd["feature"].str.split("_", n=4, expand=True)
split_df.columns = ["compartment", "feature_group", "measurement", "organelle", "rest"]
valid_organelles = {"Actin", "Mitochondria", "Hoechst", "ER", "PM"}
split_df["organelle"] = split_df["organelle"].where(
    split_df["organelle"].isin(valid_organelles), "Other"
)
healthy_vs_failing_drug_x_emd = pd.concat(
    [
        healthy_vs_failing_drug_x_emd,
        split_df[["compartment", "feature_group", "measurement", "organelle"]],
    ],
    axis=1,
)

# Save the results to parquet file
output_file = output_dir / "healthy_vs_failing_drug_x_emd.parquet"
healthy_vs_failing_drug_x_emd.to_parquet(output_file, index=False)

# Print the final DataFrame
healthy_vs_failing_drug_x_emd.head()


# ## Compute EMD comparing failing drug_x cells to the failing DMSO cells (reference)

# In[8]:


# Reference query to split the shuffled data
reference_query = '(Metadata_cell_type == "failing") & (Metadata_treatment == "DMSO")'

# Compute the baseline null range for EMD (5th and 95th percentiles)
fvfd_threshold_lower, fvfd_threshold_upper = compute_null_emd_range(
    reference_df=failing_DMSO_df,
    comparison_df=failing_drug_x_df,
    reference_query=reference_query,
    num_permutations=20,
)

# Create one-row dataframe to save threshold range to CSV
fvfd_row = pd.DataFrame(
    [
        {
            "Reference": "failing_DMSO",
            "Comparison": "failing_drug_x",
            "EMD_threshold_lower": fvfd_threshold_lower,
            "EMD_threshold_upper": fvfd_threshold_upper,
        }
    ]
)

# Append to existing CSV file (no header, append mode)
print("Writing to:", output_emd_thresholds.resolve())
fvfd_row.to_csv(output_emd_thresholds, mode="a", index=False, header=False)

# Print the threshold range
print(f"EMD threshold range: [{fvfd_threshold_lower:.4f}, {fvfd_threshold_upper:.4f}]")


# In[9]:


# Compute the signed EMD for each feature
failing_vs_failing_drug_x_emd = compute_signed_emd_per_feature(
    reference_df=failing_DMSO_df, comparison_df=failing_drug_x_df
)

# Print the results
print(failing_vs_failing_drug_x_emd.shape)
failing_vs_failing_drug_x_emd.sort_values("signed_emd", ascending=False).head()


# ### Split feature column into parts

# In[10]:


# Split the 'feature' column into new columns as requested (not as a function)
split_df = failing_vs_failing_drug_x_emd["feature"].str.split("_", n=4, expand=True)
split_df.columns = ["compartment", "feature_group", "measurement", "organelle", "rest"]
valid_organelles = {"Actin", "Mitochondria", "Hoechst", "ER", "PM"}
split_df["organelle"] = split_df["organelle"].where(
    split_df["organelle"].isin(valid_organelles), "Other"
)
failing_vs_failing_drug_x_emd = pd.concat(
    [
        failing_vs_failing_drug_x_emd,
        split_df[["compartment", "feature_group", "measurement", "organelle"]],
    ],
    axis=1,
)

# Save the results to parquet file
output_file = output_dir / "failing_vs_failing_drug_x_emd.parquet"
failing_vs_failing_drug_x_emd.to_parquet(output_file, index=False)

# Print the final DataFrame
failing_vs_failing_drug_x_emd.head()


# ## Compute EMD comparing healthy DMSO cells to the failing DMSO cells (reference)

# In[11]:


# Reference query to split the shuffled data
reference_query = '(Metadata_cell_type == "failing") & (Metadata_treatment == "DMSO")'

# Compute the baseline null range for EMD (5th and 95th percentiles)
hvf_threshold_lower, hvf_threshold_upper = compute_null_emd_range(
    reference_df=failing_DMSO_df,
    comparison_df=healthy_DMSO_df,
    reference_query=reference_query,
    num_permutations=20,
)

# Create one-row dataframe to save threshold range to CSV
hvf_row = pd.DataFrame(
    [
        {
            "Reference": "failing_DMSO",
            "Comparison": "healthy_DMSO",
            "EMD_threshold_lower": hvf_threshold_lower,
            "EMD_threshold_upper": hvf_threshold_upper,
        }
    ]
)

# Append to existing CSV file (no header, append mode)
print("Writing to:", output_emd_thresholds.resolve())
hvf_row.to_csv(output_emd_thresholds, mode="a", index=False, header=False)

# Print the threshold range
print(f"EMD threshold range: [{hvf_threshold_lower:.4f}, {hvf_threshold_upper:.4f}]")


# In[12]:


# Compute the signed EMD for each feature
failing_vs_healthy_DMSO_emd = compute_signed_emd_per_feature(
    reference_df=failing_DMSO_df, comparison_df=healthy_DMSO_df
)

# Print the results
print(failing_vs_healthy_DMSO_emd.shape)
failing_vs_healthy_DMSO_emd.sort_values("signed_emd", ascending=False).head()


# ### Split feature column into parts

# In[13]:


# Split the 'feature' column into new columns as requested (not as a function)
split_df = failing_vs_healthy_DMSO_emd["feature"].str.split("_", n=4, expand=True)
split_df.columns = ["compartment", "feature_group", "measurement", "organelle", "rest"]
valid_organelles = {"Actin", "Mitochondria", "Hoechst", "ER", "PM"}
split_df["organelle"] = split_df["organelle"].where(
    split_df["organelle"].isin(valid_organelles), "Other"
)
failing_vs_healthy_DMSO_emd = pd.concat(
    [
        failing_vs_healthy_DMSO_emd,
        split_df[["compartment", "feature_group", "measurement", "organelle"]],
    ],
    axis=1,
)

# Save the results to parquet file
output_file = output_dir / "failing_vs_healthy_DMSO_emd.parquet"
failing_vs_healthy_DMSO_emd.to_parquet(output_file, index=False)

# Print the final DataFrame
failing_vs_healthy_DMSO_emd.head()

