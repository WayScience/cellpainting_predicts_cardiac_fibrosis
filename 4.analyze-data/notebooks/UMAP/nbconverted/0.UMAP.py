#!/usr/bin/env python
# coding: utf-8

# ## Extract UMAP embeddings for CFReT CP Features

# In[1]:


import glob
import pathlib
import pandas as pd
import umap

from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Set constants
umap_random_seed = 1234
umap_n_components = 2

output_dir = pathlib.Path("results")


# In[3]:


# Set input paths
data_dir = pathlib.Path("..", "..", "..", "3.process-cfret-features", "data")

# Select only the feature selected files
file_suffix = "*sc_norm_fs_cellprofiler.csv.gz"

# Obtain file paths for all feature selected plates
fs_files = glob.glob(f"{data_dir}/{file_suffix}")
fs_files


# In[4]:


# Load feature data into a dictionary, keyed on plate name
cp_dfs = {x.split("/")[-1]: pd.read_csv(x) for x in fs_files}

# Print out useful information about each dataset
print(cp_dfs.keys())
[cp_dfs[x].shape for x in cp_dfs]


# In[5]:


# Fit UMAP features per dataset and save
for plate_name in cp_dfs:
    # Make sure to reinitialize UMAP instance per plate
    umap_fit = umap.UMAP(
        random_state=umap_random_seed,
        n_components=umap_n_components
    )
    
    # Remove NA columns
    cp_df = cp_dfs[plate_name]
    cp_df = feature_select(
        cp_df,
        operation="drop_na_columns",
        na_cutoff=0
    )
    
    # Process cp_df to separate features and metadata
    cp_features = infer_cp_features(cp_df)
    meta_features = infer_cp_features(cp_df, metadata=True)
    
    # Fit UMAP and convert to pandas DataFrame
    embeddings = pd.DataFrame(
        umap_fit.fit_transform(cp_df.loc[:, cp_features]),
        columns=[f"UMAP{x}" for x in range(0, umap_n_components)]
    )
    print(embeddings.shape)
    
    # Combine with metadata
    cp_umap_with_metadata_df = pd.concat([
        cp_df.loc[:, meta_features],
        embeddings
    ], axis=1)
    
    # Generate output file and save
    output_umap_file = pathlib.Path(output_dir, f"UMAP_{plate_name}.tsv.gz")
    cp_umap_with_metadata_df.to_csv(output_umap_file, index=False, sep="\t")


# In[6]:


# Print an example output file
cp_umap_with_metadata_df.head()

