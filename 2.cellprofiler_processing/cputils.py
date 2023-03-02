"""
This collection of functions runs CellProfiler on a given set of images from a plate and will rename the .sqlite outputed to be named 
the plate that was processed.
"""

import os
import pathlib
from pathlib import Path

def rename_sqlite_file(sqlite_dir_path: pathlib.Path, plate: str):
    """Rename the .sqlite file to be {plate}.sqlite as to differentiate between the files

    Args:
        sqlite_dir_path (pathlib.Path): path to CellProfiler_output directory
        plate (str): name of the plate that was run through CellProfiler
    """
    try:
        # CellProfiler requires a name to be set in to pipeline, so regardless of plate, all sqlite files are outputed as "CFReT.sqlite"
        sqlite_file_path = pathlib.Path(f"{sqlite_dir_path}/CFReT.sqlite")
        
        new_file_name = str(sqlite_file_path).replace(sqlite_file_path.name, f"{plate}.sqlite")

        # change the file name in the directory
        Path(sqlite_file_path).rename(Path(new_file_name))
        print(f"The file is renamed to {Path(new_file_name).name}!")

    except FileNotFoundError as e:
        print(f"The CFReT.sqlite file is not found in directory. Either the pipeline wasn't ran properly or the file is already renamed.\n"
            f"{e}")
    
def run_cellprofiler(
    path_to_pipeline: str, path_to_output: str, path_to_images: str, plate_name: str
):
    """Profile batch with CellProfiler (runs segmentation and feature extraction) and rename the file after the run
    to the name of the plate

    Args:
        path_to_pipeline (str): path to the CellProfiler .cppipe file with the segmentation and feature measurement modules
        path_to_output (str): path to the output folder for the .sqlite file
        path_to_images (str): path to folder with IC images from specific plate
    """
    # runs through any files that are in the output path
    if any(
        files.name.startswith(plate_name)
        for files in pathlib.Path(path_to_output).iterdir()
    ):
        print("This plate has already been analyzed!")
        return

    # run CellProfiler on a plate that has not been analyzed yet
    command = f"cellprofiler -c -r -p {path_to_pipeline} -o {path_to_output} -i {path_to_images}"
    os.system(command)

    # rename the outputted .sqlite file to the 
    rename_sqlite_file(sqlite_dir_path=pathlib.Path(path_to_output), plate=plate_name)
