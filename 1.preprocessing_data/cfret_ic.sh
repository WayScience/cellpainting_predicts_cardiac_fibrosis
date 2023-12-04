#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data_env

# convert Jupyter notebook to Python script
jupyter nbconvert --to python --output-dir=scripts/ *.ipynb

# run Python script for IC processing 
python scripts/cfret_ic.py
