#!/usr/bin/env python
# coding: utf-8

# ## Fit a linear model on cell morphology features
# 
# We aim to determine which features are significantly impacted by drug treatment, adjusted by cell count.

# In[1]:


import pathlib
import pandas as pd

from sklearn.linear_model import LinearRegression

from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Define inputs and outputs
plate = "localhost220513100001_KK22-05-198_FactinAdjusted"  # Focusing on plate 2
file_suffix = "_sc_norm_fs_cellprofiler.csv.gz"

data_dir = pathlib.Path("..", "..", "..", "3.process-cfret-features", "data")

cp_file = pathlib.Path(data_dir, f"{plate}{file_suffix}")

output_dir = pathlib.Path("results")
output_cp_file = pathlib.Path(output_dir, f"{plate}_linear_model_cp_features.tsv")


# In[3]:


# Load data
cp_df = pd.read_csv(cp_file)

# Drop NA columns
cp_df = feature_select(
    cp_df,
    operation="drop_na_columns",
    na_cutoff=0
)

# Count number of cells per well and add to dataframe as metadata
cell_count_df = pd.DataFrame(
    cp_df.groupby("Metadata_Well").count()["Metadata_treatment"]
).reset_index()
cell_count_df.columns = ["Metadata_Well", "Metadata_cell_count_per_well"]
cp_df = cell_count_df.merge(cp_df, on=["Metadata_Well"])

# Clean the dose column to extract numeric value
cp_df = cp_df.assign(Metadata_dose_numeric=cp_df.Metadata_dose.str.strip("uM").astype(float))

# Define CellProfiler features
cp_features = infer_cp_features(cp_df)

print(f"We are testing {len(cp_features)} CellProfiler features")
print(cp_df.shape)
cp_df.head()


# ## Fit linear model

# In[4]:


# Setup linear modeling framework
variables = ["Metadata_cell_count_per_well", "Metadata_dose_numeric"]
X = cp_df.loc[:, variables]

print(X.shape)
X.head()


# In[5]:


# Fit linear model for each feature
lm_results = []
for cp_feature in cp_features:
    # Subset CP data to each individual feature (univariate test)
    cp_subset_df = cp_df.loc[:, cp_feature]

    # Fit linear model
    lm = LinearRegression(fit_intercept=True)
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
    columns=["feature", "r2_score", "cell_count_coef", "treatment_dose_coef"]
)

# Output file
lm_results.to_csv(output_cp_file, sep="\t", index=False)

print(lm_results.shape)
lm_results.head()

