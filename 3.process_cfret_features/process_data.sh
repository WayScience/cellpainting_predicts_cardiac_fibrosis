#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data_env

# convert all notebooks to python files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run python script to preprocess data before downstream analysis (ran in sequential order)
python nbconverted/0.convert_cytotable.py
python nbconverted/1.sc_quality_control.py
python nbconverted/2.single_cell_processing.py
Rscript nbconverted/3.qc_report.r
