#!/usr/bin/env python
# coding: utf-8

# # Generate single-cell crops of cells with the highest values in specific CellProfiler features based on linear model results

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd
import numpy as np
import cv2


# ## Set paths and variables

# In[2]:


path_to_images_dir = pathlib.Path("../../../1.preprocessing_data/Corrected_Images/localhost231120090001/") # Focus on plate 4

path_to_sc_dir = pathlib.Path("./sc_crops")
path_to_sc_dir.mkdir(exist_ok=True)

sc_crop_output = pathlib.Path(f"{path_to_sc_dir}/Cells_Intensity_IntegratedIntensity_Actin")
sc_crop_output.mkdir(exist_ok=True)


# ## Load in feature selected and annotated data
# 
# We merge specific metadata columns, nuclei center coordinates, and the relevant CellProfiler feature into a new data frame.

# In[3]:


# load in plate 4 feature selected data frame
plate4_fs_df = pd.read_parquet(
    pathlib.Path(
        "../../../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_feature_selected.parquet"
    ),
    columns=[
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_Cells_Number_Object_Number",
        "Metadata_treatment",
        "Metadata_heart_number",
        "Metadata_cell_type",
        "Cells_Intensity_IntegratedIntensity_Actin"
    ],
)

coords_df = pd.read_parquet(
    pathlib.Path(
        "../../../3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_annotated.parquet"
    ),
    columns=[
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_Cells_Number_Object_Number",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
    ],
)

plate4_df = pd.merge(plate4_fs_df, coords_df, on=['Metadata_Well', 'Metadata_Site', "Metadata_Cells_Number_Object_Number"])

plate4_df = plate4_df.sort_values(by='Cells_Intensity_IntegratedIntensity_Actin', ascending=False)

print(plate4_df.shape)
plate4_df.head(5)


# ## Set up dictionary to hold info to find top single-cells from the specified CellProfiler feature

# In[4]:


plate = "localhost231120090001"

# Assuming 'plate' variable holds the plate value
top_sc_dict = {}
for i, (_, row) in enumerate(plate4_df.head(5).iterrows(), start=1):
    key_to_images = f"{plate}_{row['Metadata_Well']}{row['Metadata_Site']}"
    top_sc_dict[f"top_sc_{i}"] = {
        "key_to_images": key_to_images,
        "location_center_x": row['Nuclei_Location_Center_X'],
        "location_center_y": row['Nuclei_Location_Center_Y']
    }

# Check the created dictionary
print(top_sc_dict)


# ## Generate single-cell crops

# In[5]:


for single_cell, info in top_sc_dict.items():
    # Initialize a list to store file paths
    file_paths = []

    # Create 5 different file paths with "d0" through "d4" suffixes
    for i in range(5):
        filename = f"{path_to_images_dir}/{info['key_to_images']}d{i}_illumcorrect.tiff"
        file_paths.append(filename)

    # Initialize empty lists to store the images for each channel
    blue_channel = []
    green_channel = []
    red_channel = []

    # Iterate through channels from the random well/site and assign the correct file names with the color channel
    for file_path in file_paths:
        filename = pathlib.Path(file_path).name
        if 'd0' in filename:
            blue_channel_image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            blue_channel.append(blue_channel_image)
        elif 'd1' in filename:
            green_channel_image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            green_channel.append(green_channel_image)
        elif 'd4' in filename:
            red_channel_image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            red_channel.append(red_channel_image)

    # Stack the images for each channel along the channel axis
    blue_channel_stack = np.stack(blue_channel, axis=-1)
    green_channel_stack = np.stack(green_channel, axis=-1)
    red_channel_stack = np.stack(red_channel, axis=-1)
    

    # Scale the pixel values to fit within the 16-bit range (0-65535)
    blue_channel_stack = (blue_channel_stack / np.max(blue_channel_stack) * 65535).astype(np.uint16)
    green_channel_stack = (green_channel_stack / np.max(green_channel_stack) * 65535).astype(np.uint16)
    red_channel_stack = (red_channel_stack / np.max(red_channel_stack) * 65535).astype(np.uint16)

    # Create the RGB numpy array by merging the channels
    composite_image = cv2.merge((blue_channel_stack, green_channel_stack, red_channel_stack)).astype(np.uint16)

    # Use the location_center_x and location_center_y to create a crop
    center_x = info.get("location_center_x")
    center_y = info.get("location_center_y")

    # Crop dimensions
    crop_size = 250
    half_crop = crop_size // 2

    # Ensure the center coordinates are valid
    if center_x is not None and center_y is not None:
        # Calculate crop boundaries
        top_left_x = max(int(center_x - half_crop), 0)
        top_left_y = max(int(center_y - half_crop), 0)
        bottom_right_x = min(int(center_x + half_crop), composite_image.shape[1])
        bottom_right_y = min(int(center_y + half_crop), composite_image.shape[0])

    # Perform cropping
    cropped_image = composite_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Ensure the cropped image is of size 250x250
    cropped_image = cv2.resize(cropped_image, (crop_size, crop_size))

    # Save crop images per row
    cv2.imwrite(f'{sc_crop_output}/{single_cell}_cropped_image.png', cropped_image)

