#!/usr/bin/env python
# coding: utf-8

# # Illumination Correction of CFReT Images

# ## Import Libraries

# In[1]:


import sys
import numpy as np
import pathlib
from matplotlib import pyplot as plt
from pathlib import Path
import os
import skimage

# explicit import to PyBaSiC due to not having package support
sys.path.append("./PyBaSiC/")
import pybasic

import importlib

ICutils = importlib.import_module("ICutils")


# ## Perform Illumination Correction on CFReT pilot images

# In[2]:


# Set location of input and output locations
data_path = pathlib.Path('../0.download-data/Images/')

save_path = pathlib.Path("IC_Corrected_Images")

# Set the file_extension for the images (if applicable)
file_extension = '.TIF'

# Plates to process
plates = ["localhost220512140003_KK22-05-198", "localhost220513100001_KK22-05-198_FactinAdjusted"]

# Channels to process
channels = ["d0", "d1", "d2", "d3", "d4"]

# Perform illumination correction on each channel seperately using a `for` loop:
for plate in plates:
    print("Correcting images from plate", plate)
    for channel in channels:
        # print(channel)
        print("Correcting", channel, "channel images")

    # If you want to output the flatfield and darkfield calculations, then put "output_calc=True".
    # If you would like to overwrite any existing images when running this function, set "overwrite=True"

        ICutils.run_illum_correct(
            data_path,
            save_path,
            plate=plate,
            channel=channel,
            output_calc=False,
            file_extension='.TIF',
            overwrite=True
        )

