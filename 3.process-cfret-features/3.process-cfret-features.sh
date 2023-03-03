#!/bin/bash

# convert single cell jupyter notebook to a .py file once all variable changes are made and run the file
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/ \
        --execute extract_sc_features.ipynb

# convert image jupyter notebook to a .py file once all variable changes are made and run the file
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/ \
        --execute extract_image_features.ipynb
