#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data_env

# convert all notebooks to python files into the scripts folder
jupyter nbconvert --to python --output-dir=scripts/ *.ipynb

# run python script to preprocess data before downstream analysis (ran in sequential order)
python scripts/0.convert_cytotable.py
python scripts/1.sc_quality_control.py
python scripts/2.single_cell_processing.py
