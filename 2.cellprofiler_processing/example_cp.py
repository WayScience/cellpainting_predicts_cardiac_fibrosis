#!/usr/bin/env python
# coding: utf-8

# # CellProfiler Segmentation and Feature Extraction of CFReT data

# ## Import libraries

# In[1]:


import os
import pathlib

import importlib
CPutils = importlib.import_module("cputils")


# ## Set paths for CellProfiler
# 
# The paths for the command line command can be set here. These must be `absolute paths`, so these will have to be changed from what I have set below for my local machine.

# In[2]:


# must be an absolute path to the pipeline 
path_to_pipeline = "/home/jenna/CFReT_data/2.cellprofiler_processing/CellProfiler_input/CFReT_project.cpproj"

# must be relative path for the output directory
path_to_output = "/home/jenna/CFReT_data/2.cellprofiler_processing/CellProfiler_output"

# must be relative path to the images
path_to_images_plate1 = "/home/jenna/CFReT_data/1.preprocessing-data/IC_Corrected_Images/localhost220512140003_KK22-05-198"
path_to_images_plate2 = "/home/jenna/CFReT_data/1.preprocessing-data/IC_Corrected_Images/localhost220513100001_KK22-05-198_FactinAdjusted"


# ## Run CellProfiler through command line
# 
# Below is an example of the code used in terminal to run CellProfiler on one of the plates. Since there is already a file there, it will show that it can't run.

# In[3]:


CPutils.run_cellprofiler(path_to_pipeline, path_to_output, path_to_images_plate2)


# ## Rename .sqlite file to identify the plate that was run

# ### Set variables

# In[4]:


# relative path to the directory with .sqlite files
sqlite_file_path = pathlib.Path("../CellProfiler_output/")

# name of plate from the CP run
plate1 = "localhost220512140003_KK22-05-198"
plate2 = "localhost220513100001_KK22-05-198_FactinAdjusted"


# ### Rename the file
# 
# Below example shows renaming the `CFReT.sqlite` file after the `localhost220513100001_KK22-05-198_FactinAdjusted` plate was ran.

# In[5]:


CPutils.rename_sqlite_file(sqlite_file_path, plate2)

