#!/usr/bin/env python
# coding: utf-8

# # Illumination Correction of CFReT Images

# ## Import Libraries

# In[1]:


import sys
import pathlib
import os

import timeit

sys.path.append("../utils")
import ICutils as icutils


# ## Set paths and variables

# In[2]:


# path where plates are located containing images
data_path = pathlib.Path('../0.download_data/Images/')

# path for plates with corrected images to be saved to
save_path = pathlib.Path("./Corrected_Images")
# if directory if doesn't exist, will not raise error if it already exists
os.makedirs(save_path, exist_ok=True)

# Plates to process
plates = ["localhost220512140003_KK22-05-198", "localhost220513100001_KK22-05-198_FactinAdjusted", "localhost230405150001"]

# Channels to process
channels = ["d0", "d1", "d2", "d3", "d4"]


# ## Perform Illumination Correction on CFReT pilot images
# 
# **Note:** We use the nbconverted python script to perform IC so this cell is not completed.

# In[3]:


# Perform illumination correction on each channel seperately using a `for` loop:
for plate in plates:
    print(f"Correcting images from plate {plate}")
    for channel in channels:
        print(f"Correcting {channel} channel images")

    # If you want to output the flatfield and darkfield calculations, then put "output_calc=True".
    # If you would like to overwrite any existing images when running this function, set "overwrite=True"

        icutils.run_illum_correct(
            data_path=data_path,
            save_path=save_path,
            plate=plate,
            channel=channel,
            output_calc=False,
            file_extension='.TIF',
        )

