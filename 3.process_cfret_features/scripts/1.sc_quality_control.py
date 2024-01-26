#!/usr/bin/env python
# coding: utf-8

# # Perform single-cell quality control
# 
# >**Note:** 
# > We only perform single-cell quality control filtering on **Plate 4 (localhost231120090001)**.
# 
# In this notebook, we perform single-cell quality control. To filter the single-cells, we use z-score to find outliers using the values from only one feature at a time. We use features from the AreaShape and Intensity modle to assess the quality of the segmented single-cells:
# 
# ### Assessing poor nuclei segmentation
# 
# Due to high confluence, sometimes nuclei overlap on top of each other, creating highly intense clusters within the Hoechst channel. To identify these nuclei, we use:
# 
# - **Nuclei Area:** This metric quantifies the number of pixels in a nucleus segmentation. We detect nuclei that are abnormally large, which likely indicates poor nucleus segmentation where overlapping nuclei are merged into one segmentation. 
# - **Nuclei Intensity:** This metric quantifies the total intensity of all pixels in a nucleus segmentation. In combination with abnormally large nuclei, we detect nuclei that are also highly intense, likely indicating that this a group of overlapped nuclei.
# 
# ### Assessing poor cell segmentation
# 
# Also due to high confluence, images with large, intense clusters of cells leads to errors in the segmentation algorithm that causes cells around the cluster to segmented incorrectly. When this happens, a cell is segmented around the same segmentation as the nucleus, giving it the same area which is very small for a normal cardiac fibroblast cell. To detect poorly segmented cells, we use:
# 
# - **Cells Area:** This metric quantifies the number of pixels in a cell segmentation. We detect cells that are abnormally small, which likely indicates poor cell segmentation due to high confluence clusters.

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore


# ## Set paths and variables

# In[2]:


# Directory with data
data_dir = pathlib.Path("./data/converted_profiles/")

# Directory to save cleaned data
cleaned_dir = pathlib.Path("./data/cleaned_profiles/")
cleaned_dir.mkdir(exist_ok=True)

# Load in plate 4 annotated data
plate_4_df = pd.read_parquet(f"{data_dir}/localhost231120090001_converted.parquet")

# Add plate value to plate_4 which is absent due to error
plate_4_df['Image_Metadata_Plate'] = 'localhost231120090001'

print(plate_4_df.shape)
plate_4_df.head()


# ## Identify mis-segmented nuclei from large clusters

# ### Perform z-scoring to identify nuclei outliers

# In[3]:


# Determine z-score using only Nuclei Area and Nuclei Intensity from Nuclei channel
plate_4_df["Z_Score_Area"] = zscore(plate_4_df["Nuclei_AreaShape_Area"])
plate_4_df["Z_Score_Intensity"] = zscore(
    plate_4_df["Nuclei_Intensity_IntegratedIntensity_Hoechst"]
)

# Set a threshold for considering outliers (number of standard deviations away from the mean)
outlier_threshold = 2

# Filter DataFrame for outliers
nuclei_outliers_df = plate_4_df[
    (plate_4_df["Z_Score_Area"].abs() > outlier_threshold)
    & (plate_4_df["Z_Score_Intensity"].abs() > outlier_threshold)
]

# Print outliers to assess how it detected outliers
print(nuclei_outliers_df.shape[0])
# Print the range of outliers
print("Outliers Range:")
print("Area Min:", nuclei_outliers_df['Nuclei_AreaShape_Area'].min())
print("Area Max:", nuclei_outliers_df['Nuclei_AreaShape_Area'].max())
print("Intensity Min:", nuclei_outliers_df['Nuclei_Intensity_IntegratedIntensity_Hoechst'].min())
print("Intensity Max:", nuclei_outliers_df['Nuclei_Intensity_IntegratedIntensity_Hoechst'].max())
nuclei_outliers_df[
    [
        "Nuclei_AreaShape_Area",
        "Nuclei_Intensity_IntegratedIntensity_Hoechst",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
    ]
].sort_values(by="Nuclei_AreaShape_Area", ascending=True).head()


# ### Scatter plot of single-cells based on Nuclei Area and Intensity

# In[4]:


# Set the default value to 'inlier'
plate_4_df['Outlier_Status'] = 'inlier'

# Update the 'Outlier_Status' column based on the outliers DataFrame using index
plate_4_df.loc[plate_4_df.index.isin(nuclei_outliers_df.index), 'Outlier_Status'] = 'outlier'

# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=plate_4_df,
    x="Nuclei_AreaShape_Area",
    y="Nuclei_Intensity_IntegratedIntensity_Hoechst",
    hue="Outlier_Status",
    alpha=0.6
)

# Add mean lines
plt.axvline(
    x=plate_4_df["Nuclei_AreaShape_Area"].mean(),
    color="r",
    linestyle="--",
    label=f'Mean Area: {plate_4_df["Nuclei_AreaShape_Area"].mean():.2f}',
)
plt.axhline(
    y=plate_4_df["Nuclei_Intensity_IntegratedIntensity_Hoechst"].mean(),
    color="b",
    linestyle="--",
    label=f'Mean Intensity: {plate_4_df["Nuclei_Intensity_IntegratedIntensity_Hoechst"].mean():.2f}',
)

plt.title("Scatter Plot of Nuclei Area vs. Nuclei Integrated Intensity for Plate 4")
plt.xlabel("Nuclei Area")
plt.ylabel("Nuclei Integrated Intensity (Hoechst)")
# Move legend to bottom right
plt.legend(loc='upper right', bbox_to_anchor=(0.4, 1.0), prop={'size': 10})
plt.show()


# ## Identify mis-segmented cells due to high confluence

# ### Perform z-scoring to identify cells outliers

# In[5]:


# Calculate Z-scores for the Cells Area
z_scores = zscore(plate_4_df['Cells_AreaShape_Area'])

# Set a threshold for Z-scores to find outliers below the mean (number of standard deviations away from the mean)
threshold_z = -1

# Identify outlier rows based on Z-scores greater than the mean
cells_outliers_df = plate_4_df[z_scores < threshold_z]

# Print outliers to assess how it detected outliers
print(cells_outliers_df.shape[0])
# Print the range of outliers in the 'Cells_AreaShape_Area' column
print("Outliers Range:")
print("Min:", cells_outliers_df['Cells_AreaShape_Area'].min())
print("Max:", cells_outliers_df['Cells_AreaShape_Area'].max())
cells_outliers_df[
    [
        "Cells_AreaShape_Area",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
        "Metadata_Cells_Location_Center_X",
        "Metadata_Cells_Location_Center_Y",
    ]
].sort_values(by="Cells_AreaShape_Area", ascending=False).head(10)


# ### Box plot separating single-cells by outlier status to see the distribution of cells area

# In[6]:


# Reset the default value to 'inlier'
plate_4_df['Outlier_Status'] = 'inlier'

# Update the 'Outlier_Status' column based on the outliers DataFrame using index
plate_4_df.loc[plate_4_df.index.isin(cells_outliers_df.index), 'Outlier_Status'] = 'outlier'

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(y='Outlier_Status', x='Cells_AreaShape_Area', data=plate_4_df, palette='viridis')


# Add mean lines
plt.axvline(
    x=plate_4_df["Nuclei_AreaShape_Area"].mean(),
    color="r",
    linestyle="--",
    label=f'Mean Nuclei Area: {plate_4_df["Nuclei_AreaShape_Area"].mean():.2f}',
)
# Add mean lines
plt.axvline(
    x=plate_4_df["Cells_AreaShape_Area"].mean(),
    color="b",
    linestyle="--",
    label=f'Mean Cells Area: {plate_4_df["Cells_AreaShape_Area"].mean():.2f}',
)

# Set labels and title
plt.ylabel('Outlier Status')
plt.xlabel('Cells Area')
plt.title('Box Plot of Cells AreaShape Area by Outlier Status')
plt.legend()

# Show the plot
plt.show()


# ## Remove all outliers and save cleaned data frame

# In[7]:


# Assuming nuclei_outliers_df and cells_outliers_df have the same index
outlier_indices = pd.concat([nuclei_outliers_df, cells_outliers_df]).index

# Remove rows with outlier indices from plate_4_df
plate_4_df_cleaned = plate_4_df.drop(outlier_indices)

# Remove columns from z-scoring or assigning outliers (not included for downstream analysis)
plate_4_df_cleaned = plate_4_df_cleaned.drop(
    columns=["Z_Score_Area", "Z_Score_Intensity", "Outlier_Status", "is_outlier"],
    errors="ignore",
)

# Save cleaned data for this plate
plate_name = plate_4_df['Image_Metadata_Plate'].iloc[0]
plate_4_df_cleaned.to_parquet(f"{cleaned_dir}/{plate_name}_cleaned.parquet")

# Verify the result
print(plate_4_df_cleaned.shape)
plate_4_df_cleaned.head()

