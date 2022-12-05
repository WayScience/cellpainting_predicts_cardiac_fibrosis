#!/usr/bin/env python
# coding: utf-8

# # CellProfiler Segmentation and Feature Extraction of CFReT data

# ## Import libraries

# In[1]:


import pathlib

import importlib
CPutils = importlib.import_module("cputils")


# ## Set paths for CellProfiler
# 
# The paths for the command line command can be set here. These must be `absolute paths`, so these will have to be changed from what I have set below for my local machine. 
# Paths to rename the `CFReT.sqlite` file are `relative paths`.

# In[2]:


# path to the pipeline (specifically the .cppipe file for the command line)
path_to_pipeline = "/home/jenna/CFReT_data/2.cellprofiler_processing/CellProfiler_input/CFReT_project_CL.cppipe"

# path for the output directory
path_to_output = "/home/jenna/CFReT_data/2.cellprofiler_processing/CellProfiler_output"

# path to the images for each plate
path_to_images_plate1 = "/home/jenna/CFReT_data/1.preprocessing-data/IC_Corrected_Images/localhost220512140003_KK22-05-198"
path_to_images_plate2 = "/home/jenna/CFReT_data/1.preprocessing-data/IC_Corrected_Images/localhost220513100001_KK22-05-198_FactinAdjusted"

# relative path to the directory with .sqlite files
sqlite_file_path = pathlib.Path("CellProfiler_output")

# name of plate from the CP run
plate1 = "localhost220512140003_KK22-05-198"
plate2 = "localhost220513100001_KK22-05-198_FactinAdjusted"


# ## Run CellProfiler through command line and rename the file

# ### Run plate localhost220512140003_KK22-05-198

# In[3]:


# Run plate and output sqlite file

CPutils.run_cellprofiler(path_to_pipeline, path_to_output, path_to_images_plate1, plate1)


# In[4]:


# Rename the 'CFReT.sqlite` file to `localhost220512140003_KK22-05-198.sqlite`

CPutils.rename_sqlite_file(sqlite_file_path, plate1)


# ### Run plate localhost220513100001_KK22-05-198_FactinAdjusted

# In[5]:


# Run plate localhost220513100001_KK22-05-198_FactinAdjusted

CPutils.run_cellprofiler(path_to_pipeline, path_to_output, path_to_images_plate2, plate2)


# In[6]:


# Rename the 'CFReT.sqlite` file to `localhost220513100001_KK22-05-198_FactinAdjusted.sqlite`

CPutils.rename_sqlite_file(sqlite_file_path, plate2)

