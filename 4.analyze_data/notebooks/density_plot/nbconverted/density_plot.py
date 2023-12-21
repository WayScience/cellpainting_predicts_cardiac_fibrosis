#!/usr/bin/env python
# coding: utf-8

# # Generate density plots using the number of neighbors using the whole cell segmentation 

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ## Set variables and paths

# In[2]:


# Path to directory with single-cell profiles
sc_dir = pathlib.Path("../../../3.process_cfret_features/data/single_cell_profiles").resolve(
    strict=True
)

# Output directory for figures
output_dir_fig = pathlib.Path("./figures")
output_dir_fig.mkdir(parents=True, exist_ok=True)

# Path to annotated profile for Plate 4 only
annotated_profile = pathlib.Path(f"{sc_dir}/localhost231120090001_sc_annotated.parquet").resolve(strict=True)

# Read in data frame
annotated_df = pd.read_parquet(annotated_profile)

print(annotated_df.shape)
annotated_df.head()


# ## Generate density plot with all hearts

# In[3]:


sns.set_style("whitegrid")
sns.kdeplot(
    data=annotated_df,
    x="Cells_Neighbors_NumberOfNeighbors_Adjacent",
    hue="Metadata_heart_number",
    palette=["#00FF00", "#FF00FF", "#0000FF", "#FFA500", "#FF0000", "#800080"],
    fill=True,
    common_norm=False,
)
# Set threshold line for neighbors at 4 as it is approximately when single-cells are considered "large clusters"
plt.axvline(x=4, color="red", linestyle="--", label="Number of Neighbors = 4")
plt.title(f"Density plots per heart for Plate 4")
plt.xlabel("Cells_Neighbors_NumberOfNeighbors_Adjacent")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig(
    pathlib.Path(f"{output_dir_fig}/plate_4_heart_number_neighbors.png"), dpi=500
)
plt.show()


# ## Generate density plot with only healthy hearts

# In[4]:


# Filter out only healthy hearts to generate plot
filtered_df = annotated_df[annotated_df['Metadata_cell_type'] == 'Healthy']

sns.set_style("whitegrid")
sns.kdeplot(
    data=filtered_df,
    x="Cells_Neighbors_NumberOfNeighbors_Adjacent",
    hue="Metadata_heart_number",
    palette=["#00FF00", "#0000FF"],
    fill=True,
    common_norm=False,
)
# Set threshold line for neighbors at 4 as it is approximately when single-cells are considered "large clusters"
plt.axvline(x=4, color="red", linestyle="--", label="Number of Neighbors = 4")
plt.title(f"Density plots for only healthy hearts in Plate 4")
plt.xlabel("Cells_Neighbors_NumberOfNeighbors_Adjacent")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig(
    pathlib.Path(f"{output_dir_fig}/plate_4_heart_healthy_neighbors.png"), dpi=500
)
plt.show()


# ## Generate density plot with only failing hearts

# In[5]:


# Filter out only failing hearts to generate plot
filtered_df = annotated_df[annotated_df['Metadata_cell_type'] == 'Failing']

sns.set_style("whitegrid")
sns.kdeplot(
    data=filtered_df,
    x="Cells_Neighbors_NumberOfNeighbors_Adjacent",
    hue="Metadata_heart_number",
    palette=["#FF00FF", "#FFA500", "#FF0000", "#800080"],
    fill=True,
    common_norm=False,
)
# Set threshold line for neighbors at 4 as it is approximately when single-cells are considered "large clusters"
plt.axvline(x=4, color="red", linestyle="--", label="Number of Neighbors = 4")
plt.title(f"Density plots for only failing hearts in Plate 4")
plt.xlabel("Cells_Neighbors_NumberOfNeighbors_Adjacent")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig(
    pathlib.Path(f"{output_dir_fig}/plate_4_heart_failing_neighbors.png"), dpi=500
)
plt.show()

