# 4. Analysis of CFReT Data

In this module, we perform analysis of the CFReT data to reach our goals as specified in the [main README](../README.md).

In the [notebooks folder](./notebooks/), we have two different folders for analysis:

- [linear_model](./notebooks/linear_model/): Perform linear modeling per CellProfiler feature to determine which features significant depending on the co-variates used.

- [UMAP](./notebooks/UMAP/): Generate UMAPs labeling different metadata per plate to assess if there is any clustering of morphology features.

- [histogram_plot](./notebooks/histogram_plot/): Generate histogram plot comparing the number of neighbors adjacent to each single-cell per heart number to view the distribution of neighbors across hearts.

- [ks_test](./notebooks/ks_test/): Generate ks-test volcano plot(s) comparing DMSO versus media treatment on healthy heart #7 to see how different the features are.
