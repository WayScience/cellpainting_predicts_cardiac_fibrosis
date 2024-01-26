#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the main conda environment
conda activate cfret_data_env

# convert Jupyter notebook to script
jupyter nbconvert --to script --output-dir=scripts/ *.ipynb

# run Python scripts for QC processing and report generation 
python scripts/0.whole_image_cfret_qc.py
python scripts/1.evaluate_qc.py
Rscript scripts/1.qc_report.r

