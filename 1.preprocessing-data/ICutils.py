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

def load_pybasic_data(
    channel_path: pathlib.Path,
    channel: str,
    file_extension: str = ".tif",
    verbosity: bool = True,
):
    """
    Load all images from a specified directory in preparation for pybasic illum correction

    Parameters:
    channels_path (pathlib.Path): path to directory where the all images for each channel are stored
    channel (str): The name of the channel (currently one of ["GFP", "RFP", "DAPI"])
    file_extension (str): The filename extension of the types of files to search for (default: ".tif")
    verbosity (bool): Prints out information regarding the function running through the images (default: "True")

    Returns:
        channel_images: list of ndarrays of the loaded in images
        image_files: list of strings of the paths to each image
    """
    image_files = (
        []
    )  # List that holds all of the paths to the images in the directory for a specific channel
    image_names = (
        []
    )  # List that holds the names (str) for the images that is used for saving

    images = []  # List of numpy arrays for the images that are read in

    # This for loop will run through the specified directory, find images for a specific channel (channel name is in the file name metadata),
    # save paths to a list, and then save the names to a list after stripping the file_extension
    for image_path in channel_path.iterdir():
        image_path = str(image_path)
        if channel in image_path:
            image_files.append(image_path)
            if image_path.endswith(file_extension):
                image_names.append(image_path.strip(file_extension))

    # Sorts the file paths and names into alphabetical order (puts well C at the start)
    image_files.sort()
    image_names.sort()

    # This for loop will run through the paths to the images for the specified channel and load in the images to be used to illumination correction
    # This code was sampled from the `load_data` function in PyBaSiC
    for i, image_file in enumerate(image_files):
        if verbosity and (i % 10 == 0):
            print(i, "/", len(image_files))
        images.append(skimage.io.imread(image_file))

    return images, image_names


def run_illum_correct(
    channel_path: pathlib.Path,
    save_path: pathlib.Path,
    channel: str,
    output_calc: bool = False,
    file_extension: str = ".tif",
):
    """calculates flatfield, darkfield, performs illumination correction on channel images, coverts to 8-bit and saves images into designated folder

    Parameters:
        channels_path (pathlib.Path): path to directory where the all images for each channel are stored
        save_path (pathlib.Path): path to directory where the corrected images will be saved to
        channel (str): name of channel
        output_calc (bool): outputs plots of the flatfield and darkfield function for visual review if set to 'True'
        file_extension (str): sets the file_extension for the images
    """
    # Loads in the variables returned from "load_pybasic_data" function
    images, image_names = load_pybasic_data(channel_path, channel, file_extension)

    print("Correcting", {channel})

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

    # Covert illum corrected images to uint8 for downstream analysis
    corrected_images_coverted = np.array(channel_images_corrected)
    corrected_images_coverted[
        corrected_images_coverted < 0
    ] = 0  # makes the negatives 0
    corrected_images_coverted = corrected_images_coverted / np.max(
        corrected_images_coverted
    )  # normalize the data to 0 - 1
    corrected_images_coverted = 255 * corrected_images_coverted  # Scale by 255
    corrected_images = corrected_images_coverted.astype(np.uint8)

    for i, image in enumerate(corrected_images):
        orig_file = image_names[i]
        orig_file_name = orig_file.split("/")[3]
        new_filename = pathlib.Path(f"{save_path}/{orig_file_name}_IllumCorrect.tif")

        # If the image has not been correcrted yet, then the function will save the image. If the image exists, it will skip saving.
        if not new_filename.is_file():
            skimage.io.imsave(new_filename, image)

        else:
            print(f"{new_filename.name} already exists!")