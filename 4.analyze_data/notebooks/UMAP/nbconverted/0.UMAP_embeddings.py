#!/usr/bin/env python
# coding: utf-8

# ## Extract UMAP embeddings for CFReT CP Features

# In[1]:


import glob
import pathlib
import pandas as pd
import umap
import sys

from pycytominer.cyto_utils import infer_cp_features


# ## Generate Embeddings for Whole Plates

# ### Set constant for whole plates

# In[2]:


# Set constants
umap_random_seed = 0
umap_n_components = 2

# Set embeddings directory
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)


# ### Set paths to all plates

# In[3]:


# Set input path with single-cell profiles
data_dir = pathlib.Path("../../../3.process_cfret_features/data/single_cell_profiles/")

# Select only the feature selected files
file_suffix = "*sc_feature_selected.parquet"

# Obtain file paths for all feature selected plates
fs_files = glob.glob(f"{data_dir}/{file_suffix}")
print(f"There are {len(fs_files)} feature selected files with the following paths:")
fs_files


# ### Generate dictionary with plate and data

# In[4]:


# Load feature data into a dictionary, keyed on plate name without the suffix
cp_dfs = {x.split("/")[-1].split("_sc")[0]: pd.read_parquet(x) for x in fs_files}

# Print out useful information about each dataset
print(cp_dfs.keys())
[cp_dfs[x].shape for x in cp_dfs]


# ### Fit UMAP for whole plates

# In[5]:


# Initialize a dictionary to store UMAP embeddings for each plate
umap_results = {}

# Fit UMAP features per dataset and save
for plate in cp_dfs:
    # Set output file for the UMAP
    output_umap_file = pathlib.Path(output_dir, f"UMAP_{plate}.parquet")

    # Check if the output file already exists
    if output_umap_file.exists():
        print(f"Skipping {output_umap_file.stem} as it already exists.")
        continue

    # Make sure to reinitialize UMAP instance per plate
    umap_fit = umap.UMAP(
        random_state=umap_random_seed, n_components=umap_n_components, n_jobs=1
    )

    # Set dataframe as the current plate
    cp_df = cp_dfs[plate]

    # Process cp_df to separate features and metadata
    cp_features = infer_cp_features(cp_df)
    meta_features = infer_cp_features(cp_df, metadata=True)

    # Fit UMAP and convert to pandas DataFrame
    embeddings = pd.DataFrame(
        umap_fit.fit_transform(cp_df.loc[:, cp_features]),
        columns=[f"UMAP{x}" for x in range(0, umap_n_components)],
    )
    print(f"{embeddings.shape}: {plate}")

    # Combine with metadata
    cp_umap_with_metadata_df = pd.concat(
        [cp_df.loc[:, meta_features], embeddings], axis=1
    )

    # Check and adjust dtypes dynamically
    for col in cp_umap_with_metadata_df.columns:
        if col in meta_features:
            # Try converting to numeric first (if possible), if not, keep as string
            try:
                cp_umap_with_metadata_df[col] = pd.to_numeric(
                    cp_umap_with_metadata_df[col], errors="raise", downcast="integer"
                )
            except ValueError:
                # If can't convert to numeric, keep as string
                cp_umap_with_metadata_df[col] = cp_umap_with_metadata_df[col].astype(
                    str
                )
        else:
            # For UMAP embeddings, ensure they're float
            cp_umap_with_metadata_df[col] = cp_umap_with_metadata_df[col].astype(float)

    # Store the UMAP result in the dictionary
    umap_results[plate] = cp_umap_with_metadata_df

    # Generate output file, drop unnamed column, and save
    cp_umap_with_metadata_df.to_parquet(output_umap_file, index=False)


# ## Generate embeddings for filtered data
# 
# Instead of processing all of the cells in each plate, in this section we are taking plate 3 (`localhost230405150001`), and filtering out cells based on conditions to generate UMAP embeddings.
# We will be filtering out cells as follows:
# 
# 1. Only DMSO and TGFRi (both failing and nonfailing)
# 2. Only DMSO and drug_x (both failing and nonfailing)

# In[6]:


for plate in cp_dfs:
    # Select only plate 3 and ignore the rest
    if plate != "localhost230405150001":
        continue

    # Set dataframe as the current plate
    cp_df = cp_dfs[plate]

    # Create two new dataframes that filter cells with each condition in a dictionary
    filtered_dfs = {
        "DMSO_TGFRi": cp_df.loc[
            cp_df["Metadata_treatment"].isin(["DMSO", "TGFRi"])
        ].reset_index(drop=True),
        "DMSO_drugx": cp_df.loc[
            cp_df["Metadata_treatment"].isin(["DMSO", "drug_x"])
        ].reset_index(drop=True),
    }

    # Loop through each filtered dataframe and process it
    for condition, filtered_df in filtered_dfs.items():
        # Set output file for the UMAP
        output_umap_file = pathlib.Path(output_dir, f"UMAP_{plate}_{condition}.parquet")

        # Check if the output file already exists
        if output_umap_file.exists():
            print(f"Skipping {output_umap_file.stem} as it already exists.")
            continue

        # Make sure to reinitialize UMAP instance per plate
        umap_fit = umap.UMAP(
            random_state=umap_random_seed, n_components=umap_n_components, n_jobs=1
        )

        # Process filtered_df to separate features and metadata
        cp_features = infer_cp_features(filtered_df)
        meta_features = infer_cp_features(filtered_df, metadata=True)

        # Fit UMAP and convert to pandas DataFrame
        embeddings = pd.DataFrame(
            umap_fit.fit_transform(filtered_df.loc[:, cp_features]),
            columns=[f"UMAP{x}" for x in range(0, umap_n_components)],
        )
        print(f"{embeddings.shape}: {plate} - {condition}")

        # Combine with metadata
        filtered_umap_with_metadata_df = pd.concat(
            [filtered_df.loc[:, meta_features], embeddings], axis=1
        )

        # Check and adjust dtypes dynamically
        for col in filtered_umap_with_metadata_df.columns:
            if col in meta_features:
                # Try converting to numeric first (if possible), if not, keep as string
                try:
                    filtered_umap_with_metadata_df[col] = pd.to_numeric(
                        filtered_umap_with_metadata_df[col],
                        errors="raise",
                        downcast="integer",
                    )
                except ValueError:
                    # If can't convert to numeric, keep as string
                    filtered_umap_with_metadata_df[col] = (
                        filtered_umap_with_metadata_df[col].astype(str)
                    )
            else:
                # For UMAP embeddings, ensure they're float
                filtered_umap_with_metadata_df[col] = filtered_umap_with_metadata_df[
                    col
                ].astype(float)

        # Generate output file, drop unnamed column, and save
        filtered_umap_with_metadata_df.to_parquet(output_umap_file, index=False)

