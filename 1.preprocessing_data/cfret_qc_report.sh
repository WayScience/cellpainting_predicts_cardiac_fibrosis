#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data_env

# convert Jupyter notebook to script
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python nbconverted for QC processing and report generation 
python nbconverted/0.whole_image_cfret_qc.py
python nbconverted/1.evaluate_qc.py
Rscript nbconverted/1.qc_report.r

