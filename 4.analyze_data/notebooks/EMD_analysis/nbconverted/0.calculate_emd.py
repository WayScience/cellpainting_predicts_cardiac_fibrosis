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


# ## Helper function to compute directional (signed) EMD score

# In[ ]:


def compute_signed_emd_per_feature(
    reference_df: pd.DataFrame, comparison_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute the signed Earth Mover's Distance (EMD) for each feature between two DataFrames.

    Args:
        reference_df (pd.DataFrame): The pandas DataFrame containing the "reference" data or
            what is being used as the base to compare to.
        comparison_df (pd.DataFrame): The pandas DataFrame containing the "comparison" data or
            what is being compared against the reference.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the signed EMD for each feature.
    """
    # Filter to only feature columns (non-Metadata)
    reference_features = reference_df.loc[
        :, ~reference_df.columns.str.startswith("Metadata_")
    ]
    comparison_features = comparison_df.loc[
        :, ~comparison_df.columns.str.startswith("Metadata_")
    ]

    # Only process features shared by both
    shared_features = reference_features.columns.intersection(
        comparison_features.columns
    )
    if shared_features.empty:
        raise ValueError(
            "No shared features between reference and comparison DataFrames."
        )
    not_shared = set(reference_features.columns).symmetric_difference(
        comparison_features.columns
    )
    if not_shared:
        print(f"Features not shared between reference and comparison: {not_shared}")

    # Instantiate results list
    results = []

    # Compute signed EMD for each shared feature
    for feature in shared_features:
        ref_values = reference_features[feature].dropna()
        comp_values = comparison_features[feature].dropna()

        if len(ref_values) == 0 or len(comp_values) == 0:
            continue  # skip if either side is empty

        emd = wasserstein_distance(ref_values, comp_values)
        # Determine the direction of the EMD based on the median values (com > ref = positive EMD, com < ref = negative EMD)
        direction = np.sign(np.median(comp_values) - np.median(ref_values))
        signed_emd = emd * direction

        # Append the result for this feature
        results.append({"feature": feature, "signed_emd": signed_emd})

    # Convert results to DataFrame with all features and scores
    return pd.DataFrame(results)


# ## Set output directory for calculated EMD per comparison

# In[3]:


output_dir = pathlib.Path("./emd_results")
output_dir.mkdir(parents=True, exist_ok=True)


# ## Load in data with drug_x and controls

# In[4]:


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

# In[5]:


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


# Compute the signed EMD for each feature
failing_vs_failing_drug_x_emd = compute_signed_emd_per_feature(
    reference_df=failing_DMSO_df, comparison_df=failing_drug_x_df
)

# Print the results
print(failing_vs_failing_drug_x_emd.shape)
failing_vs_failing_drug_x_emd.sort_values("signed_emd", ascending=False).head()


# ### Split feature column into parts

# In[9]:


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

# In[10]:


# Compute the signed EMD for each feature
failing_vs_healthy_DMSO_emd = compute_signed_emd_per_feature(
    reference_df=failing_DMSO_df, comparison_df=healthy_DMSO_df
)

# Print the results
print(failing_vs_healthy_DMSO_emd.shape)
failing_vs_healthy_DMSO_emd.sort_values("signed_emd", ascending=False).head()


# ### Split feature column into parts

# In[11]:


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

