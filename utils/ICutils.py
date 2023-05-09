"""
The collection of functions will load in the raw images and save their names into lists, then the images will be corrected for illumination using PyBaSiC 
and saved into a directory for further analysis. 
"""

import logging
import sys
import numpy as np
import pathlib
from matplotlib import pyplot as plt
from pathlib import Path
import os
import skimage

# explicit import to PyBaSiC due to not having package support
sys.path.append("../1.preprocessing_data/PyBaSiC/")
import pybasic


def load_pybasic_data(
    data_path: pathlib.Path,
    plate: str,
    channel: str,
    file_extension: str = ".tif",
    verbosity: bool = True,
):
    """
    Load all images from a specified directory in preparation for pybasic illum correction

    Parameters:
    data_path (pathlib.Path): Path to directory where the all folders with data per plate are stored
    plate (str): The name of the plate to correct (either "localhost220512140003_KK22-05-198" or "localhost220513100001_KK22-05-198_FactinAdjusted")
    channel (str): The name of the channel to correct(either `d0`, `d1`, `d2`, `d3`, `d4`)
    file_extension (str): The filename extension of the types of files to search for (default: ".tif")
    verbosity (bool): Prints out information regarding the function running through the images (default: "True")

    Returns:
        channel_images: list of ndarrays of the loaded in images
        image_files: list of strings of the paths to each image
    """
    # List that holds all of the paths to the images in the directory for a specific channel
    image_files = []

    # List that holds the names (str) for the images that is used for saving
    image_names = []

    # List of numpy arrays for the images that are read in
    images = []

    # This `for` loop is making a list of directories and files within the specified data_path (glob)
    for image_path in data_path.glob(f"**/*{file_extension}"):
        # Within this "glob", it will find the images from the specific plate from the name of the parent folders (e.g. plate folders)
        if plate in image_path.parent.name:
            # Finds images with the channel in the name (needs to be string to iterate through)
            if channel in str(image_path):
                # Puts all of the paths to the files in a list
                image_files.append(image_path)
                # Removes the file extension from the names from the names of the images and then puts all the names into a list
                image_names.append(image_path.stem)

    # Sorts the file paths and names into alphabetical order (puts well C at the start)
    image_files.sort()
    image_names.sort()

    # This for loop will run through the paths to the images for the specified channel and load in the images to be used to illumination correction
    # This code was sampled from the `load_data` function in PyBaSiC
    for i, image_file in enumerate(image_files):
        # if verbosity and (i % 10 == 0):
        #     print(i, "/", len(image_files))
        images.append(skimage.io.imread(image_file))

    return images, image_names


def run_illum_correct(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    plate: str,
    channel: str,
    output_calc: bool = False,
    file_extension: str = ".tif",
    overwrite: bool = False,
    verbosity: bool = False,
):
    """Calculates flatfield, darkfield, performs illumination correction on channel images, coverts to 8-bit and saves images into designated folder

    Parameters:
        data_path (pathlib.Path): Path to directory where the folders for each plate with data are stored
        save_path (pathlib.Path): Path to directory where the corrected images will be saved to
        plate (str): Name of plate
        channel (str): Name of channel
        output_calc (bool): Outputs plots of the flatfield and darkfield function for visual review if set to 'True' (default = False)
        file_extension (str): Sets the file_extension for the images
        overwrite (bool): Will save over existing images if set to 'True' (default = False)
        verbosity (bool): Will output print statements from the load_pybasic_data function
    """
    # Loads in the variables returned from "load_pybasic_data" function
    images, image_names = load_pybasic_data(
        data_path=data_path, plate=plate, channel=channel, file_extension=file_extension, verbosity=verbosity,
    )

    print("Correcting", {channel})

    try: 
        flatfield, darkfield = pybasic.basic(images, darkfield=True)

        # Optional output that displays the plots for the flatfield and darkfield calculations if set to True (default is False)
        if output_calc == True:
            plt.title("Flatfield")
            plt.imshow(flatfield)
            plt.colorbar()
            plt.show()
            plt.title("Darkfield")
            plt.imshow(darkfield)
            plt.colorbar()
            plt.show()

        # Run PyBaSiC illumination correction function on loaded in images
        channel_images_corrected = pybasic.correct_illumination(
            images_list=images,
            flatfield=flatfield,
            darkfield=darkfield,
        )
        # Convert illum corrected images to uint16 for downstream analysis
        corrected_images_converted = np.array(channel_images_corrected)
        # Makes the negatives 0
        corrected_images_converted[corrected_images_converted < 0] = 0
        # Normalize the data to 0 - 1
        corrected_images_converted = corrected_images_converted / np.max(
            corrected_images_converted
        )
        # Scale by 65535 (16-bit)
        corrected_images_converted = 65535 * corrected_images_converted
        # Convert images to 16-bit
        corrected_images = corrected_images_converted.astype(np.uint16)

        # make directory for images to be saved to if the directory does not already exist
        plate_directory = pathlib.Path(f"{save_path}/{plate}")
        if not os.path.exists(plate_directory):
            os.mkdir(plate_directory)

        # Correlate the image names to the respective image and save the images with the file name suffix '_IllumCorrect.tif'
        for i, image in enumerate(corrected_images):
            orig_file = pathlib.Path(image_names[i])
            orig_file_name = orig_file.name
            new_filename = pathlib.Path(
                f"{save_path}/{plate}/{orig_file_name}_IllumCorrect.tif"
            )

            # If set to 'True', images will be saved regardless of if the image already exists in the directory
            if overwrite == True:
                skimage.io.imsave(new_filename, image)

            # If set to 'False', and the image has not been corrected yet, then the function will save the image. If the image exists, it will skip saving.
            if overwrite == False:
                if not new_filename.is_file():
                    skimage.io.imsave(new_filename, image)

                else:
                    print(f"{new_filename.name} already exists!")
        print(
            f"All images in channel {channel} in {plate} plate have been corrected and saved"
        )
    except IndexError:
        print(f"Make sure that {data_path} is correct. This error might be occuring because the function can not access the images.")
