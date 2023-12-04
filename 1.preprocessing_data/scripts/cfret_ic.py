#!/usr/bin/env python
# coding: utf-8

# # Run illumination correction (IC) pipeline on CFReT data 

# ## Import libraries

# In[1]:


import pathlib
import pprint

import sys

sys.path.append("../utils")
import cp_parallel


# ## Set paths and variables

# In[2]:


# set the run type for the parallelization
run_name = "illum_correction"

# set path for pipeline for illumination correction
path_to_pipeline = pathlib.Path("./illum.cppipe").resolve(strict=True)

# set main output dir for all plates if it doesn't exist
output_dir = pathlib.Path("./Corrected_Images")
output_dir.mkdir(exist_ok=True)

# directory where images are located within folders
images_dir = pathlib.Path("../0.download_data/Images")

# list for plate names based on folders to use to create dictionary
plate_names = []
# iterate through 0.download_data and append plate names from folder names that contain image data from that plate
for file_path in images_dir.iterdir():
    plate_names.append(str(file_path.stem))

print("There are a total of", len(plate_names), "plates. The names of the plates are:")
for plate in plate_names:
    print(plate)


# ## Create dictionary to process all plates

# In[3]:


# create plate info dictionary with all parts of the CellProfiler CLI command to run in parallel
plate_info_dictionary = {
    name: {
        "path_to_images": pathlib.Path(list(images_dir.rglob(name))[0]).resolve(
            strict=True
        ),
        "path_to_output": pathlib.Path(f"{output_dir}/{name}"),
        "path_to_pipeline": path_to_pipeline,

    }
    for name in plate_names
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Perform IC on all plates
# 
# Note: This code cell was not ran as we prefer to perform CellProfiler processing tasks via `sh` file (bash script) which is more stable.

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=plate_info_dictionary, run_name=run_name
)

