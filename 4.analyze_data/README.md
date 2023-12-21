# 4. Analysis of CFReT Data

In this module, we perform analysis of the CFReT data to reach our goals as specified in the [main README](../README.md).

In the [notebooks folder](./notebooks/), we have two different folders for analysis:

- [linear_model](./notebooks/linear_model/): Perform linear modeling per CellProfiler feature to determine which features significant depending on the co-variates used.

- [UMAP](./notebooks/UMAP/): Generate UMAPs labeling different metadata per plate to assess if there is any clustering of morphology features.

- [density_plot](./notebooks/density_plot/): Generate density plot comparing the number of neighbors adjacent to each single-cell per heart number to view the distribution of neighbors across hearts.

## Linear Model (LM)

In this folder, we have four different types of notebooks labeled by number:

**#0**

In the `0` notebooks, we fit the linear model to each CellProfiler feature either to all the data or stratifying the data to only include cells with specific metadata values.

**#1**

In the `1` notebooks, we perform two different tasks.
One task is to find single-cell crops of the top 5 values in a given CellProfiler feature based on the LM results.
The other is to visualize the LM beta coefficients as a scatter plot.

**#2**

In the `2` notebook, we perform a power analysis to determined if we are fully powered given the number of single-cells that were segmented per plate.

**#3**

In the `3` notebook, we visualize the power analysis as a scatter plot.

## UMAP

In this folder, we have three different types of notebooks labeled by number:

**#0**

In the `0` notebook, we use the UMAP method to reduce the morphology space and output a CSV file per plate.

**#1**

In the `1` notebooks, we visualize the UMAP coordinates for either all plates, only plate 3, or only plate 4.
We label the points with different metadata based on the plate.

**#2**

In the `2` notebooks, we generate random single-cell crops per cluster of the UMAP, in which the UMAP data frame is stratified based on the range of coordinates of the cluster.
One notebook is meant for plate 3 data and the other is for plate 4 data.

## Density plot

In this folder, there is only one notebook that is used to load in the annotated data frame from plate 4, and then create a density plot with all heart number distributions on it to compare.

## Create environments

In this module, there are two different environments to create to use in the notebooks. 

- [python_analysis_env](./python_analysis_env.yml): This environment is for use in python specific notebooks.
- [R_analysis_env](./R_analysis_env.yml): This environment is for use in R specific notebooks.

You can create the environments using the code block below:

```bash
# create environment for analysis
mamba env create -f python_analysis_env.yml
```
