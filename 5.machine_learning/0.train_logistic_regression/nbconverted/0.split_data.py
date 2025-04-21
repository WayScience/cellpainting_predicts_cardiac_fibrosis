#!/usr/bin/env python
# coding: utf-8

# # Split Plate 4 data into training, testing, and holdout data

# In[1]:


import pathlib
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ## Set paths and variables

# In[2]:


# Set random state for the whole notebook to ensure reproducibility
random.seed(0)

# Path to feature selected data for plate 4
path_to_norm_data = pathlib.Path(
    "../../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_feature_selected.parquet"
).resolve(strict=True)

# Path to annotated data for plate 4
path_to_annot_data = pathlib.Path(
    "../../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_annotated.parquet"
).resolve(strict=True)

# Make directory for split data
output_dir = pathlib.Path("./data")
output_dir.mkdir(exist_ok=True)


# ## Load in feature selected data and annotated data
# 
# We want to include the number of adjacent neighbors as both a metadata and feature column.
# 
# To do this, we are loading in the annotated data, renaming the "Cells_Neighbors_NumberOfNeighbors_Adjacent" to "Metadata_Neighbors_Adjacent", and join it onto the normalized data frame.

# In[3]:


# Load in plate 4 normalized dataset
plate_4_df = pd.read_parquet(path_to_norm_data)

# Load in plate 4 annotated dataset
neighbors_df = pd.read_parquet(
    path_to_annot_data,
    columns=[
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_Nuclei_Number_Object_Number",
        "Cells_Neighbors_NumberOfNeighbors_Adjacent",
    ],
)

# Rename neighbors feature to one that includes metadata as a prefix
neighbors_df.rename(
    columns={
        "Cells_Neighbors_NumberOfNeighbors_Adjacent": "Metadata_Neighbors_Adjacent"
    },
    inplace=True,
)

# Add new metadata column of neighbors onto the normalized data frame
plate_4_df = plate_4_df.merge(
    neighbors_df,
    on=["Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number"],
    how="inner",
)

print(plate_4_df.shape)
plate_4_df.head()


# ## Split out hold out data first into two different CSVS
# 
# 1. Remove all wells from DMSO treated healthy heart #7 and remove all wells from one failing heart (random)
# 2. Remove one well from each heart (both failing and healthy)

# ### Add DMSO treated heart 7 cells to holdout df

# In[4]:


# Copy all DMSO heart #7 rows into the holdout_df
holdout_df = plate_4_df[
    (plate_4_df["Metadata_heart_number"] == 7)
    & (plate_4_df["Metadata_treatment"] == "DMSO")
]

# Check shape and output
print(
    "The shape of the holdout data frame after removing DMSO heart 7 cells is",
    holdout_df.shape,
)
holdout_df.head()


# ### Add all rows from one random failing heart to holdout df 

# In[5]:


# Add random seed to this code cell as well to avoid change the random well if this code cell if rerun
random.seed(0)

# Create a list of only the failing heart numbers
failing_heart_numbers = plate_4_df[plate_4_df["Metadata_cell_type"] == "Failing"][
    "Metadata_heart_number"
].unique()

# Select a random heart from the list of failing hearts
random_heart_number = random.choice(failing_heart_numbers)

# Find all rows from the selected failing heart to be added to the holdout data frame
random_failing_heart_rows = plate_4_df[
    (plate_4_df["Metadata_heart_number"] == random_heart_number)
    & (plate_4_df["Metadata_cell_type"] == "Failing")
]
holdout_df = pd.concat([holdout_df, random_failing_heart_rows], ignore_index=True)

# Save holdout_df as "holdout1_data" as CSV file
holdout_df.to_csv(f"{output_dir}/holdout1_data.csv", index=False)

# Check shape and output
print(
    "There were",
    random_failing_heart_rows.shape[0],
    "rows from heart number",
    random_heart_number,
)
print(
    "The shape of the holdout data frame after removing one random failing heart is",
    holdout_df.shape,
)
holdout_df.head()


# ### Generate random well per heart number and add to holdout data frame

# In[6]:


# Add random seed to this code cell as well to avoid change the random well if this code cell if rerun
random.seed(0)

# Create new df which removes the holdout data from the plate_4_df which will be used to find random wells from rest of the data
plate_4_df_filtered = plate_4_df[
    ~(
        (
            (plate_4_df["Metadata_heart_number"] == random_heart_number)
            & (plate_4_df["Metadata_cell_type"] == "Failing")
        )
        | (
            (plate_4_df["Metadata_heart_number"] == 7)
            & (plate_4_df["Metadata_treatment"] == "DMSO")
        )
    )
]

# Generate random well per heart number to add to holdout_df
random_wells = (
    plate_4_df_filtered.groupby("Metadata_heart_number")["Metadata_Well"]
    .apply(
        lambda x: random.choice(sorted(x.unique()))
    )  # Selecting a random well from sorted unique values
    .reset_index(name="Random_Metadata_Well")
)

# Filter plate_4_df_filtered based on Metadata_heart_number and Metadata_Well in random_wells
matched_rows = plate_4_df_filtered[
    (
        plate_4_df_filtered["Metadata_heart_number"].isin(
            random_wells["Metadata_heart_number"]
        )
    )
    & (plate_4_df_filtered["Metadata_Well"].isin(random_wells["Random_Metadata_Well"]))
]

# Prior to adding data into holdout_df to remove all holdout data at once, save random well data as "holdout2_data"
matched_rows.to_csv(f"{output_dir}/holdout2_data.csv", index=False)

# Add matching rows to the holdout data frame
holdout_df = pd.concat([holdout_df, matched_rows], ignore_index=True)

# Check shape and output
print("There were", matched_rows.shape[0], "rows matching the random wells per heart")
print(
    "The shape of the holdout data frame after removing a random well per heart is",
    holdout_df.shape,
)
holdout_df.head()


# ## Remove all holdout data from the plate_4_df prior to splitting

# In[7]:


# Remove all rows from holdout data (using the data frame itself was not working)
plate_4_df = plate_4_df[
    ~(
        (
            (plate_4_df["Metadata_heart_number"] == random_heart_number)
            & (plate_4_df["Metadata_cell_type"] == "Failing")
        )
        | (
            (plate_4_df["Metadata_heart_number"] == 7)
            & (plate_4_df["Metadata_treatment"] == "DMSO")
        )
        | (
            (
                plate_4_df["Metadata_heart_number"].isin(
                    random_wells["Metadata_heart_number"]
                )
            )
            & (plate_4_df["Metadata_Well"].isin(random_wells["Random_Metadata_Well"]))
        )
    )
]

print(plate_4_df.shape)
plate_4_df.head()


# ## Split remaining Plate 4 data into testing and training data

# In[8]:


# Set random state as 0 (same as the rest of the notebook)
random_state = 0

# Set the ratio of the test data to 30% (training data will be 70%)
test_ratio = 0.30

# Split the plate 4 data into training and test
training_data, testing_data = train_test_split(
    plate_4_df,
    test_size=test_ratio,
    stratify=plate_4_df[["Metadata_cell_type"]],
    random_state=random_state,
)

# View shapes and example output
print("The testing data contains", testing_data.shape[0], "single-cells.")
print("The training data contains", training_data.shape[0], "single-cells.")
testing_data.head()


# ### Save training and test data as CSVs

# In[9]:


# Save training_data as CSV file
training_data.to_csv(f"{output_dir}/training_data.csv", index=False)

# Save testing_data as CSV file
testing_data.to_csv(f"{output_dir}/testing_data.csv", index=False)

