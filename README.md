# CFReT Project

## Data

The data used in this project is a [Cell Painting assay](https://www.moleculardevices.com/applications/cell-imaging/cell-painting#gref) on [cardiac fibroblasts](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5588900/#:~:text=Definition%20by%20function,%2C%20and%20glycoproteins5%2C6.) from 3 patients that suffered from cardiac arrest. 

In this Cell Painting, there are five channels:

- `d0` (Nuclei)
- `d1` (Endoplasmic Reticulum)
- `d2` (Golgi/Plasma Membrane)
- `d3` (Mitochondria)
- `d4` (F-actin)

![Composite_Figure.png](example_figs/Composite_Figure.png)

## Goal

The goal of this project is to identify a biomarker associated with cardiac fibroblasts that indicates if the scarring caused by cardiac arrest is reversed.

## Repository Structure

| Module | Purpose | Description |
| :---- | :----- | :---------- |
| [0_download_data](0_download_data/) | Download CFReT pilot data | Download pilot images for the CFReT project |
| [1_preprocessing_data](1_preprocessing_data/) | Perform Illumination Correction (IC) | Use `BaSiCPy` to perform IC on images per channel |
| TBD | TBD | TBD |
