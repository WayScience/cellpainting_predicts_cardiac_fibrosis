#!/usr/bin/env python
# coding: utf-8

# # Perform a two-sided KS-test using the DMSO or no treatment heart #7 (healthy) cells to assess how different each feature is based on the two populations

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pycytominer.cyto_utils import infer_cp_features


# ## Define KS-test function to generate dataframe

# In[2]:


def perform_ks_test(
    df1: pd.DataFrame, df2: pd.DataFrame, cp_features: list[str]
) -> pd.DataFrame:
    """Perform and return a data frame with the KS-test results per CellProfiler feature.

    Args:
        df1 (pd.DataFrame): First data frame in the ks-test (will be represented by positive values if greater distribution)
        df2 (pd.DataFrame): First data frame in the ks-test (will be represented by negative values if greater distribution)
        cp_features (list[str]): List of strings with the names of the CellProfiler features to perform the ks-test on

    Returns:
        pd.DataFrame: Data frame containing ks-test results
    """
    ks_results = []
    for feature in cp_features:
        result = stats.ks_2samp(df1[feature], df2[feature])
        ks_statistic = result.statistic
        p_value = result.pvalue
        statistic_location = result.statistic_location
        statistic_sign = result.statistic_sign

        ks_test_result = ks_statistic * statistic_sign

        ks_results.append(
            {
                "Feature": feature,
                "KS Statistic": ks_statistic,
                "P-Value": p_value,
                "Direction": statistic_sign,
                "Stat Location": statistic_location,
                "KS_test_result": ks_test_result,
            }
        )

    ks_results_df = pd.DataFrame(ks_results)
    ks_results_df["-log10pval"] = -np.log10(ks_results_df["P-Value"])
    ks_results_df["Compartment"] = ks_results_df["Feature"].str.split("_").str[0]
    ks_results_df["Measurement"] = ks_results_df["Feature"].str.split("_").str[1]

    return ks_results_df


# ## Read in plate 4 feature selected parquet file

# In[3]:


plate4_df = pd.read_parquet(
    pathlib.Path(
        "../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_feature_selected.parquet"
    )
)

# Fill NaN values in the "Metadata_treatment" column with "None"
plate4_df['Metadata_treatment'].fillna("None", inplace=True)

# Get columns that are cp_features (will be the same for shuffled)
cp_features = infer_cp_features(plate4_df)

print(plate4_df.shape)
plate4_df.head()


# ## Shuffle the plate 4 data to get a randomized data frame to compare to the non-shuffled results

# In[4]:


# Separate features from metadata
metadata_columns = [col for col in plate4_df.columns if col.startswith('Metadata')]
feature_columns = [col for col in plate4_df.columns if col not in metadata_columns]

# Generate dataframes with respective columns
metadata_df = plate4_df[metadata_columns]
feature_df = plate4_df[feature_columns]

# Set a random state
random_state = 0

# Convert feature_df to numpy array and shuffle
feature_array = feature_df.values
for column in feature_array.T:
    np.random.seed(random_state)
    np.random.shuffle(column)

# Combine shuffled features with metadata
shuffled_feature_df = pd.DataFrame(feature_array, columns=feature_columns)
shuffled_plate4_df = pd.concat([metadata_df, shuffled_feature_df], axis=1)

print(shuffled_plate4_df.shape)
shuffled_plate4_df.head()


# ## Split shuffled and non-shuffled data based on treatment for heart #7 only

# In[5]:


# Subset the data for heart #7 with DMSO treatment
heart_7_DMSO = plate4_df[(plate4_df['Metadata_treatment'] == 'DMSO') & (plate4_df['Metadata_heart_number'] == 7)]

# Subset the data for heart #7 with no treatment
heart_7_None = plate4_df[(plate4_df['Metadata_treatment'] == 'None') & (plate4_df['Metadata_heart_number'] == 7)]

# Subset the data for heart #7 with DMSO treatment (shuffled)
shuffled_heart_7_DMSO = shuffled_plate4_df[(shuffled_plate4_df['Metadata_treatment'] == 'DMSO') & (shuffled_plate4_df['Metadata_heart_number'] == 7)]

# Subset the data for heart #7 with no treatment (shuffled)
shuffled_heart_7_None = shuffled_plate4_df[(shuffled_plate4_df['Metadata_treatment'] == 'None') & (shuffled_plate4_df['Metadata_heart_number'] == 7)]

print("Number of cells for heart #7 DMSO treatment:", heart_7_DMSO.shape[0])
print("Number of cells for heart #7 no treatment:", heart_7_None.shape[0])


# ## Perform KS-test for non-shuffled data

# In[6]:


ks_results_df = perform_ks_test(heart_7_DMSO, heart_7_None, cp_features)

print(ks_results_df.shape)
ks_results_df.head()


# ## Perform KS-test for shuffled data

# In[7]:


shuffled_ks_results_df = perform_ks_test(shuffled_heart_7_DMSO, shuffled_heart_7_None, cp_features)

print(shuffled_ks_results_df.shape)
shuffled_ks_results_df.head()


# Note: Positive values if the empirical distribution function of **DMSO** exceeds the empirical distribution function of **no treatment** at statistic_location, otherwise negative.

# ## Generate volcano plot visualizations for each dataset

# In[8]:


# Plot the volcano plot using Seaborn with legend
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ks_results_df, x='KS_test_result', y='-log10pval', hue='Measurement', style='Compartment', palette='Dark2', alpha=0.5)
plt.title('KS-Test results per feature between heart #7 DMSO and no treatment\n (non-shuffled data)')
plt.xlabel('KS Statistic including direction')
plt.ylabel('-log10(p-value)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Add a horizontal line for p-value threshold of 0. divided by the number of features
plt.axhline(y=-np.log10(0.05 / ks_results_df.shape[0]), color='red', linestyle='--')

plt.tight_layout()
plt.savefig("./figures/ks_test_healthy_dmso_none.png", dpi=500)

plt.show()


# In[9]:


# Plot the volcano plot using Seaborn with legend
plt.figure(figsize=(8, 6))
sns.scatterplot(data=shuffled_ks_results_df, x='KS_test_result', y='-log10pval', hue='Measurement', style='Compartment', palette='Dark2', alpha=0.5)
plt.title('KS-Test results per feature between heart #7 DMSO and no treatment\n (shuffled data)')
plt.xlabel('KS Statistic including direction')
plt.ylabel('-log10(p-value)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Add a horizontal line for p-value threshold of 0. divided by the number of features
plt.axhline(y=-np.log10(0.05 / shuffled_ks_results_df.shape[0]), color='red', linestyle='--')

plt.tight_layout()
plt.savefig("./figures/ks_test_healthy_dmso_none_shuffled.png", dpi=500)

plt.show()


# ## Go over the non-shuffled data to assess top differential features

# In[13]:


# Print the sorted dataframe to see the top features in both directions
sorted_df = ks_results_df.sort_values(by="KS_test_result", ascending=False)

sorted_df


# In[11]:


# Set variable for significance value to use in print statement
significance_value = -np.log10(0.05 / ks_results_df.shape[0])

# Determine the number of features above and below the significance line
above_threshold = ks_results_df[ks_results_df['-log10pval'] > significance_value].shape[0]
below_threshold = ks_results_df[ks_results_df['-log10pval'] <= significance_value].shape[0]

print(f"Number of features above significance line (-log10pval > {significance_value}):", above_threshold)
print(f"Number of features below significance line (-log10pval <= {significance_value}):", below_threshold)


# ## Extract the top differential feature for each population

# In[14]:


# Extract the first and last row Feature names
first_feature = sorted_df.iloc[0]['Feature']
last_feature = sorted_df.iloc[-1]['Feature']

# Create a list containing the two features
selected_features = [first_feature, last_feature]

# Print the list of selected features
print(selected_features)


# ## Density plots of these features comparing between treatment

# In[16]:


# Filter the plate_4_df DataFrame to include only rows with Metadata_heart_number 7
filtered_plate_df = plate4_df[plate4_df["Metadata_heart_number"] == 7]

# Generate KDE plots for each feature in selected_features with hue based on Metadata_treatment
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=filtered_plate_df, x=feature, hue="Metadata_treatment", fill=True)
    plt.title(f"KDE Plot for {feature}")
    plt.show()


# ### Look at the cell counts across treatments

# In[17]:


# Count the number of cells for each Metadata_treatment
treatment_counts = filtered_plate_df['Metadata_treatment'].value_counts()

# Print the counts
print(treatment_counts)

