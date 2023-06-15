#!/usr/bin/env python
# coding: utf-8

# # Process CFReT single cell morphology features from CellProfiler readout

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd

from pycytominer import normalize, feature_select
from pycytominer.cyto_utils import cells, output

import sys
sys.path.append("../../utils")
import extraction_utils as extract_utils


# ## Set up paths
# 
# There are two plates from the CFReT data:
# 
# 1) localhost220512140003_KK22-05-198 - contains images from well columns 1-8 (only wells that did not have phalloidin blow out)
# 2) localhost220513100001_KK22-05-198_FactinAdjusted - contains images from well columns 9-12 (wells that were adjusted for the phalloidin blow out)
# 3) localhost230405150001 - contains images of two different patient heart cells, healthy and unhealthy, that have been explored with three different types of treatments
# 
# These two plates images together make a full 384-well plate, but they are kept in their separate plates in case there are any batch effects.

# In[2]:


# Set file and directory constants
cp_dir = "../2.cellprofiler_processing"
output_dir = "data"

# Set paths for plate localhost220512140003_KK22-05-198
sql_file1 = "localhost220512140003_KK22-05-198.sqlite"
single_cell_file1 = f"sqlite:///{cp_dir}/CellProfiler_output/{sql_file1}"
platemap_file1 = "../metadata/plate_1_CFReT.csv"
sc_output_file1 = pathlib.Path(f"{output_dir}/localhost220512140003_KK22-05-198_sc_cellprofiler.csv.gz")
sc_norm_output_file1 = pathlib.Path(f"{output_dir}/localhost220512140003_KK22-05-198_sc_norm_cellprofiler.csv.gz")
sc_norm_fs_output_file1 = pathlib.Path(f"{output_dir}/localhost220512140003_KK22-05-198_sc_norm_fs_cellprofiler.csv.gz")

# Set paths for plate localhost220513100001_KK22-05-198_FactinAdjusted
sql_file2 = "localhost220513100001_KK22-05-198_FactinAdjusted.sqlite"
single_cell_file2 = f"sqlite:///{cp_dir}/CellProfiler_output/{sql_file2}"
platemap_file2 = "../metadata/plate_2_CFReT.csv"
sc_output_file2 = pathlib.Path(f"{output_dir}/localhost220513100001_KK22-05-198_FactinAdjusted_sc_cellprofiler.csv.gz")
sc_norm_output_file2 = pathlib.Path(f"{output_dir}/localhost220513100001_KK22-05-198_FactinAdjusted_sc_norm_cellprofiler.csv.gz")
sc_norm_fs_output_file2 = pathlib.Path(f"{output_dir}/localhost220513100001_KK22-05-198_FactinAdjusted_sc_norm_fs_cellprofiler.csv.gz")

# Set paths for plate localhost230405150001
sql_file3 = "localhost230405150001.sqlite"
single_cell_file3 = f"sqlite:///{cp_dir}/CellProfiler_output/{sql_file3}"
platemap_file3 = "../metadata/plate_3_CFReT.csv"
sc_output_file3 = pathlib.Path(f"{output_dir}/localhost230405150001_sc_cellprofiler.csv.gz")
sc_norm_output_file3 = pathlib.Path(f"{output_dir}/localhost230405150001_sc_norm_cellprofiler.csv.gz")
sc_norm_fs_output_file3 = pathlib.Path(f"{output_dir}/localhost230405150001_sc_norm_fs_cellprofiler.csv.gz")


# ## Set up names for linking columns between tables in the database file

# In[3]:


# Define custom linking columns between compartments
linking_cols = {
    "Per_Cytoplasm": {
        "Per_Cells": "Cytoplasm_Parent_Cells",
        "Per_Nuclei": "Cytoplasm_Parent_Nuclei",
    },
    "Per_Cells": {"Per_Cytoplasm": "Cells_Number_Object_Number"},
    "Per_Nuclei": {"Per_Cytoplasm": "Nuclei_Number_Object_Number"},
}


# ## Load and view platemaps file per plate

# ### Plate localhost220512140003_KK22-05-198

# In[4]:


# Load platemap file for plate localhost220512140003_KK22-05-198
platemap_df1 = pd.read_csv(platemap_file1)
platemap_df1

print(platemap_df1.shape)
platemap_df1.head()


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[5]:


# Load platemap file for plate localhost220513100001_KK22-05-198_FactinAdjusted
platemap_df2 = pd.read_csv(platemap_file2)
platemap_df2

print(platemap_df2.shape)
platemap_df2.head()


# ### Plate localhost230405150001

# In[6]:


# Load platemap file for plate localhost230405150001
platemap_df3 = pd.read_csv(platemap_file3)
platemap_df3

print(platemap_df3.shape)
platemap_df3.head()


# ## Set up `SingleCells` class from Pycytominer

# ### Plate localhost220512140003_KK22-05-198

# In[7]:


# Instantiate SingleCells class
sc1 = cells.SingleCells(
    sql_file=single_cell_file1,
    compartments=["Per_Cells", "Per_Cytoplasm", "Per_Nuclei"],
    compartment_linking_cols=linking_cols,
    image_table_name="Per_Image",
    strata=["Image_Metadata_Well", "Image_Metadata_Plate"],
    merge_cols=["ImageNumber"],
    image_cols="ImageNumber",
    load_image_data=True
)


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[8]:


# Instantiate SingleCells class
sc2 = cells.SingleCells(
    sql_file=single_cell_file2,
    compartments=["Per_Cells", "Per_Cytoplasm", "Per_Nuclei"],
    compartment_linking_cols=linking_cols,
    image_table_name="Per_Image",
    strata=["Image_Metadata_Well", "Image_Metadata_Plate"],
    merge_cols=["ImageNumber"],
    image_cols="ImageNumber",
    load_image_data=True
)


# ### Plate localhost230405150001

# In[9]:


# Instantiate SingleCells class
sc3 = cells.SingleCells(
    sql_file=single_cell_file3,
    compartments=["Per_Cells", "Per_Cytoplasm", "Per_Nuclei"],
    compartment_linking_cols=linking_cols,
    image_table_name="Per_Image",
    strata=["Image_Metadata_Well", "Image_Metadata_Plate"],
    merge_cols=["ImageNumber"],
    image_cols="ImageNumber",
    load_image_data=True
)


# ## Merge single cells

# ### Plate localhost220512140003_KK22-05-198

# In[10]:


# Merge single cells across compartments
anno_kwargs1 = {"join_on": ["Metadata_well_position", "Image_Metadata_Well"]}

sc_df1 = sc1.merge_single_cells(
    platemap=platemap_df1,
    **anno_kwargs1,
)

# Save level 2 data as a csv
output(sc_df1, sc_output_file1)

print(sc_df1.shape)
sc_df1.head()


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[11]:


# Merge single cells across compartments
anno_kwargs2 = {"join_on": ["Metadata_well_position", "Image_Metadata_Well"]}

sc_df2 = sc2.merge_single_cells(
    platemap=platemap_df2,
    **anno_kwargs2,
)

# Save level 2 data as a csv
output(sc_df2, sc_output_file2)

print(sc_df2.shape)
sc_df2.head()


# ### Plate localhost230405150001

# In[12]:


# Merge single cells across compartments
anno_kwargs3 = {"join_on": ["Metadata_well_position", "Image_Metadata_Well"]}

sc_df3 = sc3.merge_single_cells(
    platemap=platemap_df3,
    **anno_kwargs3,
)

# Save level 2 data as a csv
output(sc_df3, sc_output_file3)

print(sc_df3.shape)
sc_df3.head()


# ## Normalize data

# ### Plate localhost220512140003_KK22-05-198

# In[13]:


# Normalize single cell data and write to file
normalize_sc_df1 = normalize(
    sc_df1,
    method="standardize"
)

output(normalize_sc_df1, sc_norm_output_file1)

print(normalize_sc_df1.shape)
normalize_sc_df1.head()


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[14]:


# Normalize single cell data and write to file
normalize_sc_df2 = normalize(
    sc_df2,
    method="standardize"
)

output(normalize_sc_df2, sc_norm_output_file2)

print(normalize_sc_df2.shape)
normalize_sc_df2.head()


# ### Plate localhost230405150001

# In[15]:


# Normalize single cell data and write to file
normalize_sc_df3 = normalize(
    sc_df3,
    method="standardize"
)

output(normalize_sc_df3, sc_norm_output_file3)

print(normalize_sc_df3.shape)
normalize_sc_df3.head()


# ## Feature selection

# ### Plate localhost220512140003_KK22-05-198

# In[16]:


feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "blocklist",
]

feature_select_norm_sc_df1 = feature_select(
    normalize_sc_df1,
    operation=feature_select_ops
)

output(feature_select_norm_sc_df1, sc_norm_fs_output_file1)

print(feature_select_norm_sc_df1.shape)
feature_select_norm_sc_df1.head()


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[17]:


feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "blocklist",
]

feature_select_norm_sc_df2 = feature_select(
    normalize_sc_df2,
    operation=feature_select_ops
)

output(feature_select_norm_sc_df2, sc_norm_fs_output_file2)

print(feature_select_norm_sc_df2.shape)
feature_select_norm_sc_df2.head()


# ### Plate localhost230405150001

# In[18]:


feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "blocklist",
]

feature_select_norm_sc_df3 = feature_select(
    normalize_sc_df3,
    operation=feature_select_ops
)

output(feature_select_norm_sc_df3, sc_norm_fs_output_file3)

print(feature_select_norm_sc_df3.shape)
feature_select_norm_sc_df3.head()


# ## Add single cell counts for all `csv.gz` files

# In[19]:


# Plate localhost220512140003_KK22-05-198
extract_utils.add_sc_count_metadata(sc_output_file1)
extract_utils.add_sc_count_metadata(sc_norm_output_file1)
extract_utils.add_sc_count_metadata(sc_norm_fs_output_file1)

# Plate localhost220513100001_KK22-05-198_FactinAdjusted
extract_utils.add_sc_count_metadata(sc_output_file2)
extract_utils.add_sc_count_metadata(sc_norm_output_file2)
extract_utils.add_sc_count_metadata(sc_norm_fs_output_file2)


# ---
# 
# ### Visualize basic count statistics

# ### Plate localhost220512140003_KK22-05-198

# In[20]:


sc_df1.Metadata_dose.value_counts()


# In[21]:


pd.crosstab(sc_df1.Metadata_dose, sc_df1.Metadata_Well)


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[22]:


sc_df2.Metadata_dose.value_counts()


# In[23]:


pd.crosstab(sc_df2.Metadata_dose, sc_df2.Metadata_Well)

