#!/usr/bin/env python
# coding: utf-8

# # Process image features from CellProfiler readout - Only Factin_Adjusted
# 
# After discussion, we want to look into the Factin_Adjusted plate as it contains the better protocols for the actin channel so we are more interested in this data.
# In this notebook, we create two different CSV outputs, one with the image features (e.g., Granularity, Texture, etc.) and one for the image quality metrics (e.g. blur).

# ## Import Libraries

# In[1]:


import pathlib
import pandas as pd

from pycytominer import annotate
from pycytominer.cyto_utils import output

import sys
sys.path.append("../../utils")
import extraction_utils as extract_utils


# ## Set up paths to CellProfiler directory and outputs

# In[2]:


# Set file and directory constants
cp_dir = "../2.cellprofiler_processing"
output_dir = "data"


# ## Set paths to sqlite files

# In[3]:


# Set name and path of .sqlite file and path to metadata
sql_file = "localhost220513100001_KK22-05-198_FactinAdjusted.sqlite"
single_cell_file = f"sqlite:///{cp_dir}/CellProfiler_output/{sql_file}"
# plate_2 is synonymous with the Factin_Adjusted plate
platemap_file = "metadata/plate_2_CFReT.csv"
image_table_name = "Per_Image"

# Set paths with name for outputted data
image_quality_output_file = pathlib.Path(f"{output_dir}/image_quality_Factin_Adjusted.csv.gz")
image_features_output_file = pathlib.Path(f"{output_dir}/image_features_Factin_Adjusted.csv.gz")


# ## Set variables for extracting image measurements

# In[4]:


# These image feature categories are based on the measurement modules ran in the CellProfiler pipeline 
# Texture not included as it was added post running the plate originally
image_feature_categories = ["Image_Correlation", "Image_Granularity", "Image_Intensity", "Image_Texture"]
image_quality_category = ["Image_ImageQuality"]
image_cols=["ImageNumber", "Image_Count_Cells", "Image_Count_Cytoplasm", "Image_Count_Nuclei"]
strata=["Image_Metadata_Well", "Image_Metadata_Plate"]


# ## Load and view platemap file

# In[5]:


# Load platemap file
platemap_df = pd.read_csv(platemap_file)
platemap_df.head()


# ## Load in sqlite file

# In[6]:


image_df = extract_utils.load_sqlite_as_df(single_cell_file, image_table_name)

print(image_df.shape)
image_df.head()


# ## Extract image features from sqlite file

# In[7]:


image_features_df = extract_utils.extract_image_features(image_feature_categories, image_df, image_cols, strata)

print(image_features_df.shape)
image_features_df.head()


# ## Extract image quality from sqlite file

# In[8]:


image_quality_df = extract_utils.extract_image_features(image_quality_category, image_df, image_cols, strata)

print(image_quality_df.shape)
image_quality_df.head()


# ## Merge platemap metadata with extracted image features and image quality

# In[9]:


## Uses pycytominer annotate functionality to merge the platemap and image features and reorder the dataframe
image_features_df = annotate(
    profiles=image_features_df,
    platemap=platemap_df,
    join_on=["Metadata_well_position", "Image_Metadata_Well"],
    output_file="none",
)

image_quality_df = annotate(
    profiles=image_quality_df,
    platemap=platemap_df,
    join_on=["Metadata_well_position", "Image_Metadata_Well"],
    output_file="none",
)


# ## Save image features data frame as `csv.gz` file

# In[10]:


# Save image feature data as a csv
output(image_features_df, image_features_output_file)

print(image_features_df.shape)
image_features_df.head()


# ## Save image quality dataframe as `csv.gz` file

# In[11]:


# Save image feature data as a csv
output(image_quality_df, image_quality_output_file)

print(image_quality_df.shape)
image_quality_df.head()


# ## View info of the dataframe

# In[12]:


image_features_df.info()


# In[13]:


image_quality_df.info()

