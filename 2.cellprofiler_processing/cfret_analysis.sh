#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data

# convert the notebook into a python and run the file
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/ \
        --execute cfret_analysis.ipynb
