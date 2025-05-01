#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the R environment
conda activate r_analysis_cfret

# convert all notebooks to python files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run script to generate figure(s)
Rscript nbconverted/generate_figure_drugx.r
