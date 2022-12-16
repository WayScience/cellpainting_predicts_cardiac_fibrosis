#!/usr/bin/env python
# coding: utf-8

# # Process CFReT single cell morphology features from CellProfiler readout

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd

from pycytominer import normalize
from pycytominer.cyto_utils import cells, output


# ## Set up paths

# In[2]:


# Set file and directory constants
cp_dir = "../2.cellprofiler_processing"
output_dir = "data"

# Set paths for plate localhost220512140003_KK22-05-198
sql_file1 = "localhost220512140003_KK22-05-198.sqlite"
single_cell_file1 = f"sqlite:///{cp_dir}/CellProfiler_output/{sql_file1}"
platemap_file1 = "metadata/plate_1_CFReT.csv"
sc_output_file1 = pathlib.Path(f"{output_dir}/localhost220512140003_KK22-05-198_sc_cellprofiler.csv.gz")
sc_norm_output_file1 = pathlib.Path(f"{output_dir}/localhost220512140003_KK22-05-198_sc_norm_cellprofiler.csv.gz")

# Set paths for plate localhost220513100001_KK22-05-198_FactinAdjusted
sql_file2 = "localhost220513100001_KK22-05-198_FactinAdjusted.sqlite"
single_cell_file2 = f"sqlite:///{cp_dir}/CellProfiler_output/{sql_file2}"
platemap_file2 = "metadata/plate_2_CFReT.csv"
sc_output_file2 = pathlib.Path(f"{output_dir}/localhost220513100001_KK22-05-198_FactinAdjusted_sc_cellprofiler.csv.gz")
sc_norm_output_file2 = pathlib.Path(f"{output_dir}/localhost220513100001_KK22-05-198_FactinAdjusted_sc_norm_cellprofiler.csv.gz")


# ## Set up names for linking columns between tables in the database file

# In[3]:


# Define custom linking columns between compartments
linking_cols = {
    "Per_Cytoplasm": {
        "Per_Cells": "Cytoplasm_Parent_Cells",
        "Per_Nuclei": "Cytoplasm_Parent_OrigNuclei",
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


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[5]:


# Load platemap file for plate localhost220513100001_KK22-05-198_FactinAdjusted
platemap_df2 = pd.read_csv(platemap_file2)
platemap_df2


# ## Set up `SingleCells` class from Pycytominer

# ### Plate localhost220512140003_KK22-05-198

# In[6]:


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

# In[7]:


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


# ## Merge single cells

# ### Plate localhost220512140003_KK22-05-198

# In[8]:


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

# In[9]:


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


# ## Normalize data

# ### Plate localhost220512140003_KK22-05-198

# In[10]:


# Normalize single cell data and write to file
normalize_sc_df1 = normalize(
    sc_df1,
    method="standardize"
)

output(normalize_sc_df1, sc_norm_output_file1)

print(normalize_sc_df1.shape)
normalize_sc_df1.head()


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[11]:


# Normalize single cell data and write to file
normalize_sc_df2 = normalize(
    sc_df2,
    method="standardize"
)

output(normalize_sc_df2, sc_norm_output_file2)

print(normalize_sc_df2.shape)
normalize_sc_df2.head()


# ---
# 
# ### Visualize basic count statistics

# ### Plate localhost220512140003_KK22-05-198

# In[12]:


sc_df1.Metadata_dose.value_counts()


# In[13]:


pd.crosstab(sc_df1.Metadata_dose, sc_df1.Metadata_Well)


# ### Plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[14]:


sc_df2.Metadata_dose.value_counts()


# In[15]:


pd.crosstab(sc_df2.Metadata_dose, sc_df2.Metadata_Well)

