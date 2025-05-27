#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pandas as pd

from sklearn.linear_model import LinearRegression

from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Define inputs and outputs
plate = "localhost220512140003_KK22-05-198"  # Focusing on plate 1 (with same disease etiology heart)
file_suffix = "_sc_feature_selected.parquet"

data_dir = pathlib.Path(
    "../../../3.process_cfret_features/data/single_cell_profiles"
).resolve(strict=True)

cp_file = pathlib.Path(data_dir, f"{plate}{file_suffix}")

output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
output_cp_file = pathlib.Path(output_dir, f"{plate}_linear_model_dose_count.tsv")


# In[3]:


# Load data
cp_df = pd.read_parquet(cp_file)

# Filter to only the cells from heart #3
cp_df = cp_df[cp_df["Metadata_heart_number"] == 3]

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


# In[4]:


# Prepare the data for the linear model
variables = ["Metadata_cell_count_per_well", "Metadata_dose"]
X = cp_df.loc[:, variables]

print(X.shape)
X.head()


# In[5]:


# Initialize the linear regression model
model = LinearRegression()

# Prepare a dictionary to store the results
feature_contributions = {}

# Fit a linear model for each CellProfiler feature
for feature in cp_features:
    y = cp_df[feature]  # Target variable
    model.fit(X, y)  # Fit the model

    # Store the coefficients for cell count and dose
    feature_contributions[feature] = {
        "r2_score": model.score(X, y),
        "cell_count_coef": model.coef_[0],
        "dose_coef": model.coef_[1],
        "intercept": model.intercept_,
    }

# Convert the results to a DataFrame for easier analysis
contributions_df = pd.DataFrame.from_dict(feature_contributions, orient="index")
contributions_df.reset_index(inplace=True)
contributions_df.rename(columns={"index": "Feature"}, inplace=True)

print(contributions_df.shape)
contributions_df.head()

# Save the results to a file
contributions_df.to_csv(output_cp_file, sep="\t", index=False)

print(f"Linear model results saved to {output_cp_file}")


# In[6]:


# Small exploration visualization
contributions_df.plot(x="cell_count_coef", y="dose_coef", kind="scatter")

