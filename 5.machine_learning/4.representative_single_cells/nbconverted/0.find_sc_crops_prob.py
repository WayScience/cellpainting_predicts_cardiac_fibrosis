#!/usr/bin/env python
# coding: utf-8

# # Find one top representative single-cell crop per heart and treatment
# 
# We are identifying one representative single-cell per heart and treatment from plate 3 only for the following:
# 
# 1. healthy + DMSO
# 2. failing + DMSO
# 3. failing + drug_x
# 
# We first filter the plate 3 data frame to only include isolated cells (0 cell neighbors adjacent) and then filter out any single-cell that is too close to an edge (based on crop_size).
# 

# ## Import libraries

# In[1]:


import pathlib
from pprint import pprint

import cv2
import pandas as pd
from typing import List


# ## Define functions for notebook

# In[2]:


# Function for formatting min/max row data frames into dictionaries
def create_sc_dict(dfs: List[pd.DataFrame], names: List[str]) -> dict:
    """Format lists of data frames and names into a dictionary with all relevant metadata to find single-cell images.

    Args:
        dfs (List[pd.DataFrame]): List of data frames each containing a single cell and relevant metadata.
        names (List[str]): List of names corresponding to the data frames.

    Returns:
        dict: Dictionary containing info relevant for finding single-cell crops.
    """
    sc_dict = {}
    for df, name in zip(dfs, names):
        for i, (_, row) in enumerate(df.iterrows()):
            key = f"{name}"
            sc_dict[key] = {
                "plate": row["Metadata_Plate"],
                "well": row["Metadata_Well"],
                "site": row["Metadata_Site"],
                "location_center_x": row["Metadata_Nuclei_Location_Center_X"],
                "location_center_y": row["Metadata_Nuclei_Location_Center_Y"],
            }
    return sc_dict


# In[3]:


# Function for generating and saving single-cell crops per channel as PNGs
def generate_sc_crops(
    sc_dict: dict,
    images_dir: pathlib.Path,
    output_img_dir: pathlib.Path,
    crop_size: int,
) -> None:
    """Using a dictionary with single-cell metadata info per image set, single-cell crops per channel are generated
    and saved as PNGs in an image set folder.

    Args:
        sc_dict (dict): Dictionary containing info relevant for finding single-cell crops.
        images_dir (pathlib.Path): Directory where illumination corrected images are found.
        output_img_dir (pathlib.Path): Main directory to save each image set single-cell crops
        crop_size (int): Size of the box in pixels (example: setting crop_size as 250 will make a 250x250 pixel crop around the single-cell center coordinates)
    """
    for key, info in sc_dict.items():
        # Initialize a list to store file paths for every image set
        file_paths = []

        # Create file paths with well, site, and channel
        for i in range(5):  # Update the range to start from 0 and end at 4
            filename = f"{images_dir}/{info['plate']}_{info['well']}{info['site']}d{i}_illumcorrect.tiff"
            file_paths.append(filename)

            # Read the image
            channel_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # Use the location_center_x and location_center_y to create a crop
            center_x = info.get("location_center_x")
            center_y = info.get("location_center_y")

            # Crop dimensions (including crop_size)
            half_crop = crop_size // 2

            # Ensure the center coordinates are valid
            if center_x is not None and center_y is not None:
                # Calculate crop boundaries
                top_left_x = max(int(center_x - half_crop), 0)
                top_left_y = max(int(center_y - half_crop), 0)
                bottom_right_x = min(int(center_x + half_crop), channel_image.shape[1])
                bottom_right_y = min(int(center_y + half_crop), channel_image.shape[0])

                # Perform cropping
                cropped_channel = channel_image[
                    top_left_y:bottom_right_y, top_left_x:bottom_right_x
                ]

                # Ensure the cropped image is of size 250x250
                cropped_channel = cv2.resize(cropped_channel, (crop_size, crop_size))

                # Make directory for the key to keep all channels for an image in one folder
                key_dir = pathlib.Path(f"{output_img_dir}/{key}")
                key_dir.mkdir(exist_ok=True, parents=True)

                # Save the cropped image with single_cell and channel information
                output_filename = str(pathlib.Path(f"{key_dir}/{key}_d{i}_cropped.png"))
                cv2.imwrite(output_filename, cropped_channel)


# ## Set paths and variables

# In[4]:


# Images directory for plate 3
images_dir = pathlib.Path(
    "../../1.preprocessing_data/Corrected_Images/localhost230405150001"
).resolve(strict=True)

# Output dir for cropped images
output_img_dir = pathlib.Path("./sc_crops")
output_img_dir.mkdir(exist_ok=True)

# Define the size of the cropping box (250x250 pixels)
crop_size = 250

# Create open list for one row data frames for each top feature per channel per cell type
list_of_dfs = []

# Create open list of names to assign each data frame in a list relating to the feature, channel, and cell type
list_of_names = []


# ## Load in plate 3 data probability data

# In[5]:


# Load in plate 3 probability data (CSV file)
plate3_prob_df = pd.read_csv(
    pathlib.Path(
        "../3.assess_generalizability/prob_data/combined_plate_3_predicted_proba.csv"
    )
)

# Filter for only rows from final model
plate3_prob_df = plate3_prob_df[plate3_prob_df["model_type"] == "final"]

# Load in annotated dataframe for plate 3 to extract neighbors
annot_df = pd.read_parquet(
    pathlib.Path(
        "../../3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_annotated.parquet"
    ),
    columns=[
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_Nuclei_Number_Object_Number",
        "Cells_Neighbors_NumberOfNeighbors_Adjacent",
    ],
)

plate3_prob_df = plate3_prob_df.merge(
    annot_df,
    on=["Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number"],
    how="inner",
)

plate3_prob_df.rename(
    columns={
        "Cells_Neighbors_NumberOfNeighbors_Adjacent": "Metadata_Number_of_Cells_Neighbors_Adjacent"
    },
    inplace=True,
)

print(plate3_prob_df.shape)
plate3_prob_df.head()


# ## Filter for isolated single cells

# In[6]:


# Filter the DataFrame directly
filtered_plate3_prob_df = plate3_prob_df[
    (plate3_prob_df["Metadata_Number_of_Cells_Neighbors_Adjacent"].isin([0]))
    & (plate3_prob_df["Metadata_Nuclei_Location_Center_X"] > crop_size // 2)
    & (
        plate3_prob_df["Metadata_Nuclei_Location_Center_X"]
        < (plate3_prob_df["Metadata_Nuclei_Location_Center_X"].max() - crop_size // 2)
    )
    & (plate3_prob_df["Metadata_Nuclei_Location_Center_Y"] > crop_size // 2)
    & (
        plate3_prob_df["Metadata_Nuclei_Location_Center_Y"]
        < (plate3_prob_df["Metadata_Nuclei_Location_Center_Y"].max() - crop_size // 2)
    )
]

print(filtered_plate3_prob_df.shape)
filtered_plate3_prob_df.head()


# ## Confirm that we have cells are each cell type we want to find a representative crop for

# In[7]:


filtered_plate3_prob_df[
    (filtered_plate3_prob_df["Metadata_cell_type"] == "Healthy")
    & (filtered_plate3_prob_df["Metadata_treatment"] == "DMSO")
].shape[0]

filtered_plate3_prob_df.head()


# In[8]:


filtered_plate3_prob_df[
    (filtered_plate3_prob_df["Metadata_cell_type"] == "Failing")
    & (filtered_plate3_prob_df["Metadata_treatment"] == "DMSO")
].shape[0]


# In[9]:


filtered_plate3_prob_df[
    (filtered_plate3_prob_df["Metadata_cell_type"] == "Failing")
    & (filtered_plate3_prob_df["Metadata_treatment"] == "drug_x")
].shape[0]


# ## Representative cell for healthy + DMSO

# In[10]:


# Get data frame with a random representative healthy DMSO cell predicted with high confidence to be healthy
top_healthy_dmso = (
    filtered_plate3_prob_df[
        (filtered_plate3_prob_df["Metadata_cell_type"] == "Healthy")
        & (filtered_plate3_prob_df["Metadata_treatment"] == "DMSO")
    ]
    .sort_values("Healthy_probas", ascending=False)
    .sample(1, random_state=0)
)

top_healthy_dmso = top_healthy_dmso[
    [
        "Healthy_probas",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_heart_number",
        "Metadata_cell_type",
    ]
]


# Append the DataFrame and its name to the lists
list_of_dfs.append(top_healthy_dmso)
list_of_names.append("top_healthy_dmso")

print(top_healthy_dmso.shape)
top_healthy_dmso


# ## Representative single-cell from failing DMSO

# In[11]:


# Get data frame with a random representative failing DMSO cell predicted with high confidence to be failing
top_failing_dmso = (
    filtered_plate3_prob_df[
        (filtered_plate3_prob_df["Metadata_cell_type"] == "Failing")
        & (filtered_plate3_prob_df["Metadata_treatment"] == "DMSO")
    ]
    .sort_values("Healthy_probas", ascending=True)
    .sample(1, random_state=0)
)

top_failing_dmso = top_failing_dmso[
    [
        "Healthy_probas",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_heart_number",
        "Metadata_cell_type",
    ]
]


# Append the DataFrame and its name to the lists
list_of_dfs.append(top_failing_dmso)
list_of_names.append("top_failing_dmso")

print(top_failing_dmso.shape)
top_failing_dmso


# ## Representative single-cell for failing drug-x

# In[12]:


# Get data frame with a random representative failing drug_x cell predicted with high confidence to be healthy
top_failing_drug_x = (
    filtered_plate3_prob_df[
        (filtered_plate3_prob_df["Metadata_cell_type"] == "Failing")
        & (filtered_plate3_prob_df["Metadata_treatment"] == "drug_x")
    ]
    .sort_values("Healthy_probas", ascending=False)
    .sample(1, random_state=0)
)

top_failing_drug_x = top_failing_drug_x[
    [
        "Healthy_probas",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_heart_number",
        "Metadata_cell_type",
    ]
]


# Append the DataFrame and its name to the lists
list_of_dfs.append(top_failing_drug_x)
list_of_names.append("top_failing_drug_x")

print(top_failing_drug_x.shape)
top_failing_drug_x


# In[13]:


sc_dict = create_sc_dict(dfs=list_of_dfs, names=list_of_names)

# Check the created dictionary for the first two items
pprint(list(sc_dict.items())[:2], indent=4)


# In[14]:


generate_sc_crops(
    sc_dict=sc_dict,
    images_dir=images_dir,
    output_img_dir=output_img_dir,
    crop_size=crop_size,
)

