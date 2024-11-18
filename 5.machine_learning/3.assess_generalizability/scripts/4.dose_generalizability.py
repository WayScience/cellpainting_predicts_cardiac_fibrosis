#!/usr/bin/env python
# coding: utf-8

# # Assess generalizability of the model by using drug dose curve data
# 
# **NOTE:** For assess generalizability based on a drug dose response, we will be using Plates 1 and 2, split by heart number (all failing hearts).

# ## Import libraries

# In[1]:


import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from joblib import load
from sklearn.metrics import precision_recall_curve

sys.path.append("../utils")
from eval_utils import generate_confusion_matrix_df, generate_f1_score_df
from training_utils import get_X_y_data


# ## Set paths and variables

# In[2]:


# Directory with plate datasets
data_dir = pathlib.Path("../3.process_cfret_features/data/single_cell_profiles")

# Directory with models
models_dir = pathlib.Path("./models")

# Directory for model figures output
fig_dir = pathlib.Path("./figures")
fig_dir.mkdir(exist_ok=True)

# Directory for probability data to be saved
prob_dir = pathlib.Path("./prob_data")
prob_dir.mkdir(exist_ok=True)

# Load in each model individually
final_model = load(
    pathlib.Path(f"{models_dir}/log_reg_fs_plate_4_final_downsample.joblib")
)
shuffled_model = load(
    pathlib.Path(f"{models_dir}/log_reg_fs_plate_4_shuffled_downsample.joblib")
)


# ## Load in Plate 4 fs data to extract column names to filter from the other plates

# In[3]:


# Load in Plate 4 normalized feature selected data metadata (used with model) to get the feature columns to filter the plate data
parquet_metadata = pq.read_metadata(
    pathlib.Path(f"{data_dir}/localhost231120090001_sc_feature_selected.parquet")
)

# Get the column names from the metadata
all_column_names = parquet_metadata.schema.names

# Filter out the column names that start with "Metadata_"
model_column_names = [
    col for col in all_column_names if not col.startswith("Metadata_")
]

print(len(model_column_names))
print(model_column_names)


# ## Load in Plates 1 and 2, concat vertically, and drop any rows where there are NaNs in the feature columns from the model

# In[4]:


# Load in Plate 1 and 2 data -> concat
plate_1_df = pd.read_parquet(
    pathlib.Path(f"{data_dir}/localhost220512140003_KK22-05-198_sc_normalized.parquet")
)
plate_2_df = pd.read_parquet(
    pathlib.Path(
        f"{data_dir}/localhost220513100001_KK22-05-198_FactinAdjusted_sc_normalized.parquet"
    )
)

# Concat separate parts of the same plate together
concatenated_df = pd.concat([plate_1_df, plate_2_df], axis=0)

# Drop rows with NaN values in feature columns that the model uses
concatenated_df = concatenated_df.dropna(subset=model_column_names)

print(concatenated_df.shape)
concatenated_df.head()


# ## Filter the concat data to only include metadata and filtered feature columns

# In[5]:


# Extract metadata columns from the plate
metadata_columns = [col for col in concatenated_df.columns if col.startswith("Metadata_")]

# Extract feature columns that don't start with "Metadata_"
feature_columns = [col for col in concatenated_df.columns if not col.startswith("Metadata_")]

# Filter columns in data frame to only include those in the model
filtered_feature_columns = [
    col for col in concatenated_df.columns if col in model_column_names
]

# Filter the DataFrame to keep only the desired columns
concatenated_df = concatenated_df[metadata_columns + filtered_feature_columns]

concatenated_df


# ## Create a dictionary with concat dataframe splitting the data by the heart number

# In[6]:


# Split the plate data into different data frames with different data for applying the model to

# Define a dictionary
plate_1_2_dfs_dict = {}

# Filter the DataFrame to a data frame per treatment
three_df = concatenated_df[concatenated_df["Metadata_heart_number"] == 3]
eight_df = concatenated_df[concatenated_df["Metadata_heart_number"] == 8]
nine_df = concatenated_df[concatenated_df["Metadata_heart_number"] == 9]

# Add each DataFrame to the dictionary with a corresponding key
plate_1_2_dfs_dict["heart_3"] = {"data_df": three_df}
plate_1_2_dfs_dict["heart_8"] = {"data_df": eight_df}
plate_1_2_dfs_dict["heart_9"] = {"data_df": nine_df}


# ## Extract final model predicted probabilities for each heart number

# In[7]:


# Create an empty DataFrame to store the results
combined_prob_df = pd.DataFrame()

for model_path in models_dir.iterdir():
    if model_path.is_dir() or model_path.suffix != ".joblib":
        continue  # Skip directories or files that are not model files
    
    model_type = model_path.stem.split("_")[5]  # Get the model type

    for data, info in plate_1_2_dfs_dict.items():
        # Ensure that the file is named the correct data split
        data_split = data
        print(f"Extracting {model_type} probabilities from {data} data...")

        # Load in model to apply to datasets
        model = load(model_path)

        # Load in label encoder
        le = load(
            pathlib.Path("./encoder_results/label_encoder_log_reg_fs_plate_4.joblib")
        )

        # Get unique cell types and their corresponding encoded values
        unique_labels = le.classes_
        encoded_values = le.transform(unique_labels)

        # Create a dictionary mapping encoded values to original labels
        label_dict = dict(zip(encoded_values, unique_labels))

        # Load in data frame associated with the data split
        data_df = info["data_df"].reset_index(drop=True)

        # Load in X data to get predicted probabilities
        X, _ = get_X_y_data(df=data_df, label="Metadata_heart_number")

        # Predict class probabilities for morphology feature data
        predicted_probs = model.predict_proba(X)

        # Storing probabilities in a pandas DataFrame
        prob_df = pd.DataFrame(predicted_probs, columns=model.classes_)

        # Update column names in prob_df using the dictionary and add suffix "_probas"
        prob_df.columns = [label_dict[col] + "_probas" for col in prob_df.columns]

        # Add a new column called predicted_label for each row
        prob_df["predicted_label"] = prob_df.apply(
            lambda row: row.idxmax()[:-7], axis=1
        )

        # Select metadata columns from the data
        metadata_columns = data_df.filter(like="Metadata_")

        # Combine metadata columns with predicted probabilities DataFrame based on index
        prob_df = prob_df.join(metadata_columns)

        # Add a new column for model_type
        prob_df["model_type"] = model_type

        # Append the probability DataFrame to the combined DataFrame
        combined_prob_df = pd.concat([combined_prob_df, prob_df], ignore_index=True)

# Save combined prob data
combined_prob_df.to_csv(f"{prob_dir}/combined_plates_1_2_predicted_proba.csv", index=False)

