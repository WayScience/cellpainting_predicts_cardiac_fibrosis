#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data

# convert notebook to python file into the scripts folder
jupyter nbconvert --to python --output-dir=scripts/ illumcorrect_data.ipynb

# execute and output time it takes to complete python file
time python scripts/illumcorrect_data.py
