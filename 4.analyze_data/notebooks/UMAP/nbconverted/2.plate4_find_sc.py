#!/usr/bin/env python
# coding: utf-8

# ## Generate single-cell crop images based on the plate 4 UMAP coordinates

# ## Import libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib
import cv2


# ## Set paths and variables

# In[2]:


umap_df = pd.read_csv(pathlib.Path("./results/UMAP_localhost231120090001_sc_feature_selected.tsv.gz"), sep='\t')

images_dir = pathlib.Path("../../../1.preprocessing_data/Corrected_Images/localhost231120090001/")

plate = "localhost231120090001"

output_dir = pathlib.Path("plate4_umap_crops")
output_dir.mkdir(parents=True,exist_ok=True)

# Output dir for composite and cropped images
output_img_dir = pathlib.Path("./images")
output_img_dir.mkdir(exist_ok=True)


# ## Find single-cells per cluster
# 
# - right_cluster
#   - x_min, x_max = 7.5, 10 
#   - y_min, y_max = 2.5, 4 
# - bottom_cluster
#   - x_min, x_max = -1.5, 1.5
#   - y_min, y_max = -5, -2 
# - big_middle_cluster
#   - x_min, x_max = -2.5, 6.25 
#   - y_min, y_max = -2.5, 3.75  
# - left_top_cluster
#   - x_min, x_max = -1.25, 1.25 
#   - y_min, y_max = 3.75, 7 
# - right_top_cluster
#   - x_min, x_max = 1.25, 3
#   - y_min, y_max = 3.75, 5.5

# In[6]:


# Define boundaries for the region you want to filter
x_min, x_max = 1.25, 3
y_min, y_max = 3.75, 5.5
name_of_cluster = "right_top_cluster"

# Filter dataframe based on specified region
filtered_df = umap_df[
    (umap_df['UMAP0'] >= x_min) & (umap_df['UMAP0'] <= x_max) &
    (umap_df['UMAP1'] >= y_min) & (umap_df['UMAP1'] <= y_max)
]

# Randomly sample 3 rows from the filtered dataframe
random_sample = filtered_df.sample(n=3, random_state=0) 

# Save random sample df for each cluster to review later
random_sample.to_csv(f"{output_dir}/{name_of_cluster}.csv")

# Loop through each row in the sampled dataframe
for index, row in random_sample.iterrows():
    # Initialize a list to store file paths
    file_paths = []
    # Assuming the format is "{plate}_{well}{site}"
    well = row['Metadata_Well']  # Replace 'well' with the correct column name
    site = row['Metadata_Site']  # Replace 'site' with the correct column name
    
    # Generate file paths with different suffixes "d0" through "d4"
    for i in range(5):
        filename = f"{images_dir}/{plate}_{well}{site}d{i}_illumcorrect.tiff"
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
    center_x = row['Metadata_Cells_Location_Center_X'] 
    center_y = row['Metadata_Cells_Location_Center_Y'] 

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
    cv2.imwrite(f'{output_dir}/{name_of_cluster}_cropped_image_{index}.png', cropped_image)

