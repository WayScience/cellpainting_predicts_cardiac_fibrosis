#!/usr/bin/env python
# coding: utf-8

# # Stratify to perform linear modeling on certain data

# In[1]:


import pathlib
import pandas as pd

from sklearn.linear_model import LinearRegression

from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Define inputs and outputs
plate = "localhost230405150001"  # Focusing on plate 3
file_suffix = "_sc_feature_selected.parquet"

data_dir = pathlib.Path("../../../3.process_cfret_features/data/single_cell_profiles")

data_df = pd.read_parquet(pathlib.Path(data_dir, f"{plate}{file_suffix}"))

output_dir = pathlib.Path("results")
output_cp_file = pathlib.Path(output_dir, f"{plate}_linear_model_DMSO_failing_healthy.tsv")

print(data_df.shape)
data_df.head()


# ## Stratify data

# In[3]:


# Filter by failing hearts and specific treatments
specific_type = ["DMSO"]
specific_cell_types = ["failing", "healthy"]

filtered_df = data_df[
    (data_df['Metadata_treatment'].isin(specific_type)) &
    (data_df['Metadata_cell_type'].isin(specific_cell_types))
]

# Drop NA columns
cp_df = feature_select(
    filtered_df,
    operation="drop_na_columns",
    na_cutoff=0
)

# Count number of cells per well and add to dataframe as metadata
cell_count_df = pd.DataFrame(
    cp_df.groupby("Metadata_Well").count()["Metadata_treatment"]
).reset_index()
cell_count_df.columns = ["Metadata_Well", "Metadata_cell_count_per_well"]
cp_df = cell_count_df.merge(cp_df, on=["Metadata_Well"])

# Define CellProfiler features
cp_features = infer_cp_features(cp_df)

print(f"We are testing {len(cp_features)} CellProfiler features")
print(cp_df.shape)
cp_df.head()


# ## Fit linear model

# In[5]:


# Setup linear modeling framework -> in plate 3 we are looking at the treatments or cell type
variables = ["Metadata_cell_count_per_well", "Metadata_cell_type"]
X = cp_df.loc[:, variables]

print(X.shape)
X.head()


# In[6]:


# Set the variables and treatments used for LM
variables = ["Metadata_cell_count_per_well", "Metadata_cell_type"]
treatments_to_select = ["failing", "healthy"]

# Select rows with specific treatment values
selected_rows = X[X["Metadata_cell_type"].isin(treatments_to_select)]

# Create dummy variables
dummies = pd.get_dummies(selected_rows["Metadata_cell_type"])

# Concatenate dummies with the selected rows DataFrame
X = pd.concat([selected_rows, dummies], axis=1)

# Drop the original treatment column
X.drop("Metadata_cell_type", axis=1, inplace=True)

print(X.shape)
X.head()


# In[7]:


# Fit linear model for each feature
lm_results = []
for cp_feature in cp_features:
    # Create a boolean mask to filter rows with the specified treatments
    mask = cp_df["Metadata_cell_type"].isin(treatments_to_select)

    # Apply the mask to Subset CP data to each individual feature (univariate test)
    cp_subset_df = cp_df.loc[mask, cp_feature]

    # Fit linear model
    lm = LinearRegression()
    lm_result = lm.fit(X=X, y=cp_subset_df)
    
    # Extract Beta coefficients
    # (contribution of feature to X covariates)
    coef = lm_result.coef_
    
    # Estimate fit (R^2)
    r2_score = lm.score(X=X, y=cp_subset_df)
    
    # Add results to a growing list
    lm_results.append([cp_feature, r2_score] + list(coef))

# Convert results to a pandas DataFrame
lm_results = pd.DataFrame(
    lm_results,
    columns=["feature", "r2_score", "cell_count_coef", "failing_coef", "healthy_coef"]
)

# Output file
lm_results.to_csv(output_cp_file, sep="\t", index=False)

print(lm_results.shape)
lm_results.head()

