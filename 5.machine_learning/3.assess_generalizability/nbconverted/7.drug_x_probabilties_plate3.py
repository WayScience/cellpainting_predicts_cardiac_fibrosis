#!/usr/bin/env python
# coding: utf-8

# # Generate histograms with the predicted probabilities

# In[1]:


import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

import pyarrow.parquet as pq

import sys

sys.path.append("../../utils")
from training_utils import get_X_y_data


# In[2]:


# Load in final model to apply data to
model = load(
    "../0.train_logistic_regression/models/log_reg_fs_plate_4_final_downsample.joblib"
)

# Set output directory for figure
fig_dir = pathlib.Path("./figures")
fig_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


# Load in the plate 3 normalized data
plate3_df = pd.read_parquet(
    pathlib.Path(
        "../../3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_normalized.parquet"
    )
)

# Load in the model columns to filter the dataframe
cols = pq.read_schema(
    "../../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_feature_selected.parquet"
).names
model_columns = [col for col in cols if not col.startswith("Metadata_")]

print(len(model_columns))

# Filter down the feature columns in plate 3 dataframe with just metadata and model columns
plate3_df_model_ready = plate3_df.loc[
    :,
    plate3_df.columns.str.startswith("Metadata_")
    | plate3_df.columns.isin(model_columns),
].dropna(subset=model_columns)

# Check output
print(plate3_df_model_ready.shape)
plate3_df_model_ready.head()


# In[4]:


# Get in X data to get predicted probabilities
X, _ = get_X_y_data(df=plate3_df_model_ready, label="Metadata_cell_type")

# Apply the model to the dataframe
plate3_df_model_ready["predicted_class"] = model.predict(X)
plate3_df_model_ready["predicted_class"] = plate3_df_model_ready["predicted_class"].map(
    {0: "failing", 1: "healthy"}
)

# Extract relevant columns for matching
prob_df = plate3_df_model_ready[
    [
        "Metadata_cell_type",
        "Metadata_heart_number",
        "Metadata_treatment",
        "predicted_class",
    ]
]

# Display the result
print(prob_df.shape)
prob_df.head()


# In[5]:


# Calculate proportion of healthy predictions per heart number and treatment
proportion_healthy = (
    prob_df.groupby(["Metadata_heart_number", "Metadata_treatment"])["predicted_class"]
    .apply(lambda x: (x == "healthy").mean())
    .reset_index(name="proportion_healthy")
)

# Save as CSV file
proportion_healthy.to_csv(
    "./prob_data/proportion_healthy_plate3_grouped.csv", index=False, header=True
)

# Print output shape and display
print(proportion_healthy.shape)
proportion_healthy


# In[6]:


# Drop the "TGFRi" treatment
proportion_healthy = proportion_healthy[
    proportion_healthy["Metadata_treatment"] != "TGFRi"
]

# Define a custom palette for treatments
custom_palette = {
    "DMSO": "#BA5A31",
    "drug_x": "#E7298A",
}


# Plot the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(
    data=proportion_healthy,
    x="Metadata_heart_number",
    y="proportion_healthy",
    hue="Metadata_treatment",
    palette=custom_palette,
    dodge=True,
    width=0.7,
)

# Add axis labels
plt.xlabel("Heart Number")
plt.ylabel("Proportion of cells predicted as healthy")
plt.tight_layout()

# Save figure as png and pdf
plt.savefig(
    f"{fig_dir}/predicted_healthy_proportion_per_heart_number.pdf",
    bbox_inches="tight",
)
plt.savefig(
    f"{fig_dir}/predicted_healthy_proportion_per_heart_number.png",
    bbox_inches="tight",
)

plt.show()

