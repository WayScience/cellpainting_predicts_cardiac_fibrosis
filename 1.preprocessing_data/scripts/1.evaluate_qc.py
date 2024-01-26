#!/usr/bin/env python
# coding: utf-8

# # Whole image quality control metric evaluation
# 
# In this notebook, we will use the outputted QC metrics to start working on developing thresholds using z-score to flag and skip images during CellProfiler illumination correction. These are currently not proven to be generalizable and will only be applied to one plate.
# 
# **Note**: We will be using Plate 4 measurements only. 
# 
# **Blur metric to detect out of focus images** -> PowerLogLogSlope
# 
# **Saturation metric to detect large smudges** -> PercentMaximal

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd

from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


# ## Set paths and load in data frame

# In[2]:


# Directory for figures to be outputted
figure_dir = pathlib.Path("./qc_figures")
figure_dir.mkdir(exist_ok=True)

# Focus on Plate 4
plate = "localhost231120090001"

# Read in CSV with all image quality metrics per image
qc_df = pd.read_csv(pathlib.Path(f"./qc_results/{plate}/Image.csv"))

# Adding 'Metadata_Well' and 'Metadata_Site' columns using filename from FileName_OrigActin (same for all channels)
qc_df['Metadata_Well'] = qc_df['FileName_OrigActin'].str.extract(r'_(\w{3})\w*\.TIF')
qc_df['Metadata_Site'] = qc_df['FileName_OrigActin'].str.extract(r'_\w{3}(\w{3})\w*\.TIF')

print(qc_df.shape)
qc_df.head()


# ## Create concat dataframe combining blur and saturation metrics from all channels

# In[3]:


# List of channels
channels = ["Actin", "DNA", "ER", "PM", "Mito"]

# Create DataFrames for each channel with all Metadata columns
Actin_df = qc_df.filter(like="Metadata_").copy()
DNA_df = Actin_df.copy()
ER_df = Actin_df.copy()
PM_df = Actin_df.copy()
Mito_df = Actin_df.copy()

# Iterate through each channel and add the specified columns
for channel in channels:
    # Add PowerLogLogSlope column
    globals()[f"{channel}_df"][f"ImageQuality_PowerLogLogSlope_Orig{channel}"] = qc_df[
        f"ImageQuality_PowerLogLogSlope_Orig{channel}"
    ]

    # Add PercentMaximal column
    globals()[f"{channel}_df"][f"ImageQuality_PercentMaximal_Orig{channel}"] = qc_df[
        f"ImageQuality_PercentMaximal_Orig{channel}"
    ]

    # Rename columns for each channel to concat
    globals()[f"{channel}_df"] = globals()[f"{channel}_df"].rename(
        columns={
            f"ImageQuality_PowerLogLogSlope_Orig{channel}": "ImageQuality_PowerLogLogSlope",
            f"ImageQuality_PercentMaximal_Orig{channel}": f"ImageQuality_PercentMaximal"
        }
    )

    # Add "Channel" column
    globals()[f"{channel}_df"]["Channel"] = channel

# Concat the channel data frames together for plotting
df = pd.concat([Actin_df, DNA_df, ER_df, PM_df, Mito_df], ignore_index=True)

print(df.shape)
df.head()


# ## Visualize blur metric
# 
# Based on the plot below, we can see there is no obvious evidence that blur is impacting the quality of the images. **We will not be filtering out image sets based on this.** Currently based on previous projects, the threshold is anything above -1.5 (meaning the values closer to 0 are more blurry).

# In[4]:


sns.set_style('whitegrid')
sns.kdeplot(data=df, x='ImageQuality_PowerLogLogSlope', hue='Channel', palette=['r', 'b', 'g', 'y', 'magenta'],fill=True, common_norm=False)
plt.title(f'Density plots per channel for {plate}')
plt.xlabel('ImageQuality_PowerLogLogSlope')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig(pathlib.Path(f"{figure_dir}/{plate}_channels_blur_density.png"), dpi=500)
plt.show()


# ## Saturation metric
# 
# For saturation metrics, we are looking for:
# 
# - Smudged images (usually seen in `DNA` channel regardless of stain)
# - Specifically for the CFReT project, some cells will start clustering where they are overlapping, causing issues during segmentation. Based on the below threshold method, we can see that intensely clustered cells are easily identified in the `PM` channel.
# 
# This means that we will only be setting threshold for outliers that are detected from the DNA and PM channels in the CellProfiler IC pipeline.

# In[5]:


summary_statistics = df["ImageQuality_PercentMaximal"].describe()
print(summary_statistics)


# In[6]:


# Calculate Z-scores for the column
z_scores = zscore(df['ImageQuality_PercentMaximal'])

# Set a threshold for Z-scores (adjust as needed for number of standard deviations away from the mean)
threshold_z = 2

# Identify outlier rows based on Z-scores greater than as to identify whole images with abnormally high saturated pixels
outliers = df[abs(z_scores) > threshold_z]

# Remove any outliers detected in other channels (currently we have concluded that those other channels don't detect artifacts)
outliers = outliers[(outliers['Channel'] == 'DNA') | (outliers['Channel'] == 'PM')]

# Save outliers data frame to view in report
outliers.to_csv("./qc_results/qc_outliers.csv")

print(outliers.shape)
print(outliers['Channel'].value_counts())
outliers.sort_values(by='ImageQuality_PercentMaximal')


# We will use the lowest value in the calculated outliers to determine the threshold where any if one image in a set has a value higher then the whole image set is skipped during IC.
# 
# **We will be using 0.28 as the threshold.**

# In[10]:


# Reset the default value to 'inlier'
df['Outlier_Status'] = 'inlier'

# Update the 'Outlier_Status' column based on the outliers DataFrame using index
df.loc[df.index.isin(outliers.index), 'Outlier_Status'] = 'outlier'

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(y='Outlier_Status', x='ImageQuality_PercentMaximal', data=df, palette='viridis')


# Add mean lines
plt.axvline(
    x=df["ImageQuality_PercentMaximal"].mean(),
    color="r",
    linestyle="--",
    label=f'Mean Nuclei Area: {df["ImageQuality_PercentMaximal"].mean():.2f}',
)
plt.axvline(
    x=0.28,
    color="b",
    linestyle="--",
    label='Threshold of Outliers: 0.28',
)

# Set labels and title
plt.ylabel('Outlier Status')
plt.xlabel('Percent Maximal')
plt.title('Box Plot of Percent Maximal by Outlier Status')
plt.legend()

# Show the plot
plt.show()

