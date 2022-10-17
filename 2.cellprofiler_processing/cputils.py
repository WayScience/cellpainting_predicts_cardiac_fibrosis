"""
This collection of functions runs CellProfiler on a given set of images from a plate and will rename the .sqlite outputed to be named 
the plate that was processed.
"""

import os
import pathlib
from pathlib import Path


def run_cellprofiler(path_to_pipeline: str, path_to_output: str, path_to_images: str):
    """Profile batch with CellProfiler (runs segmentation and feature extraction)

    Args:
        path_to_pipeline (str): path to the CellProfiler .cppipe file with the segmentation and feature measurement modules
        path_to_output (str): path to the output folder for the .sqlite file
        path_to_images (str): path to folder with IC images from specific plate
    """
    # run CellProfiler on a plate
    command = f"cellprofiler -c -r -p {path_to_pipeline} -o {path_to_output} -i {path_to_images}"
    os.system(command)


def rename_sqlite_file(sqlite_file_path: pathlib.Path, plate: str):
    """Rename the .sqlite file to be {plate}.sqlite as to differentiate between the files

    Args:
        sqlite_file_path (pathlib.Path): path to CellProfiler_output directory
        plate (str): name of the plate that was run through CellProfiler
    """
    for sqlite_files in sqlite_file_path.iterdir():
        # the CellProfiler pipeline hardcodes the .sqlite file name, so all files outputted are called "CFReT.sqlite"
        if "CFReT.sqlite" in str(sqlite_files):
            # change the file name to include the plate name
            new_file_name = str(sqlite_files).replace(
                sqlite_files.name, f"{plate}.sqlite"
            )
            # change the file name in the directory
            Path(sqlite_files).rename(Path(new_file_name))
