#!/usr/bin/env python
# coding: utf-8

# # CellProfiler Segmentation and Feature Extraction of CFReT data

# ## Import libraries

# In[1]:


import pathlib
import sys

sys.path.append("../utils/")
import cputils as cp_utils


# ## Set paths and variables

# In[2]:


# path to pipeline and output to use for all plates
path_to_pipeline = pathlib.Path(
    "./CellProfiler_input/CFReT_project_CL.cppipe"
).resolve()
path_to_output = pathlib.Path("./CellProfiler_output/").resolve()

plate_info_dictionary = {
    "localhost220512140003_KK22-05-198": {
        "path_to_images": pathlib.Path(
            "../1.preprocessing_data/Corrected_Images/localhost220512140003_KK22-05-198/"
        ).resolve(),
    },
    "localhost220513100001_KK22-05-198_FactinAdjusted": {
        "path_to_images": pathlib.Path(
            "../1.preprocessing_data/Corrected_Images/localhost220513100001_KK22-05-198_FactinAdjusted"
        ).resolve(),
    },
    "localhost230405150001": {
        "path_to_images": pathlib.Path(
            "../1.preprocessing_data/Corrected_Images/localhost230405150001/"
        ).resolve(),
    },
}


# ## Run CellProfiler analysis on all plates
# 
# **Note:** This will run each of the plates in sequential order, with the first plate in the dictionary running first. The SQLite file output will be renamed to the plate that was ran after the run finishes before starting the next plate.

# In[3]:


# run through each plate sequentially with each set of paths based on dictionary
for plate, info in plate_info_dictionary.items():
    path_to_images = info["path_to_images"]
    print(f"Running analysis on {plate}!")

    # run analysis pipeline
    cp_utils.run_cellprofiler(
        path_to_pipeline=path_to_pipeline,
        path_to_output=path_to_output,
        path_to_images=path_to_images,
        # name each SQLite file after plate name
        sqlite_name=plate,
        analysis_run=True,
    )

