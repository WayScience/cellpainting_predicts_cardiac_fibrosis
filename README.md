# CFReT Project

## Data

The data used in this project is a modified [Cell Painting assay](https://www.moleculardevices.com/applications/cell-imaging/cell-painting#gref) on [cardiac fibroblasts](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5588900/#:~:text=Definition%20by%20function,%2C%20and%20glycoproteins5%2C6.). 

In this modified Cell Painting, there are five channels:

- `d0` (Nuclei)
- `d1` (Endoplasmic Reticulum)
- `d2` (Golgi/Plasma Membrane)
- `d3` (Mitochondria)
- `d4` (F-actin)

![Composite_Figure.png](example_figs/Composite_Figure.png)

## Plate maps

We applied this modified Cell Painting assay using the following plate design for the first two plates:

- **localhost220512140003_KK22-05-198**

![localhost220512140003_KK22-05-198_platemap_figure.png](./metadata/platemap_figures/localhost220512140003_KK22-05-198_platemap_figure.png)

- **localhost220513100001_KK22-05-198_FactinAdjusted**

![localhost220513100001_KK22-05-198_FactinAdjusted_platemap_figure.png](./metadata/platemap_figures/localhost220513100001_KK22-05-198_FactinAdjusted_platemap_figure.png)

For the third plate, we are using the following plate design:

- **localhost230405150001**

![localhost230405150001_platemap_figure.png](./metadata/platemap_figures/localhost230405150001_platemap_figure.png)

In this plate, there are only two different patients, one with a healthy heart and one that had a failing heart. 

For the fourth plate, we used the following plate design:

- **localhost231120090001**

![localhost231120090001_platemap_figure.png](./metadata/platemap_figures/localhost231120090001_platemap_figure.png)

For this fourth plate, we are looking at different patients with the same heart failure type and patients with healthy hearts. 
We want to assess if there are morphological differences between cells that come from different patients but suffer the same type of heart failure, which is dilated cardiomyopathy.

See our [platemaps](metadata/) for more details.

## Goals

The goals of this project are:
1. To identify morphology features from cardiac fibroblasts that distinguish cardiac patients. 
2. To discover a cell morphology biomarker associated with drug treatment to reverse fibrosis scarring caused by cardiac arrest.

## Repository Structure

| Module | Purpose | Description |
| :---- | :----- | :---------- |
| [0.download_data](0.download_data/) | Download CFReT pilot data | Download pilot images for the CFReT project |
| [1.preprocessing_data](1.preprocessing_data/) | Perform Illumination Correction (IC) | We use CellProfiler to perform IC on images per channel for all plates |
| [2.cellprofiler_processing](2_cellprofiler_processing/) | Apply feature extraction pipeline | We use CellProfiler to extract hundreds of morphology features per imaging channel |
| [3.process_cfret_features](3.process_cfret_features/) | Get morphology features analysis ready | Apply cytotable and pycytominer to perform single-cell merging, annotation, normalization, and feature selection |
| [4.analyze_data](4.analyze_data/) | Analyze the single cell profiles to achieve goals listed above | Several independent analyses to describe data and test hypotheses |

## Create main CFReT conda environment

For all modules, we use one main environment for the repository, which includes all packages needed including installing CellProfiler v4.2.4 among other packages.

To create the environment, run the below code block:

```bash
# Run this command in terminal to create the conda environment
conda env create -f cfret_main_env.yml
```

**Make sure that the conda environment is activated before running notebooks or scripts:**

```bash
conda activate cfret_data
```
