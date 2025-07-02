# 4. Analysis of CFReT Data

In this module, we perform analysis of the CFReT data to reach our goals as specified in the [main README](../README.md).

In the [notebooks folder](./notebooks/), we have two different folders for analysis:

- [linear_model](./notebooks/linear_model/): Perform linear modeling per CellProfiler feature to determine which features significant depending on the co-variates used.

- [UMAP](./notebooks/UMAP/): Generate UMAPs labeling different metadata per plate to assess if there is any clustering of morphology features.

- [histogram_plot](./notebooks/histogram_plot/): Generate histogram plot comparing the number of neighbors adjacent to each single-cell per heart number to view the distribution of neighbors across hearts.

- [ks_test](./notebooks/ks_test/): Generate ks-test volcano plot(s) comparing DMSO versus media treatment on healthy heart #7 to see how different the features are.

- [EMD_analysis](./notebooks/EMD_analysis/): Perform an earth mover's distance analysis (inspired by work in the [SPACe paper](https://www.nature.com/articles/s41467-024-54264-4)) per feature in three different population comparisons using control and drug_x treated cells. Earth mover's distance is a metric that quantifies how different two populations are, which is different from performing a KS-test, which is a hypothesis test that will quantify how likely the two populations are or are not different from each other. By using EMD over KS-test, we are prioritizing evaluating the extent of the differences over evaluating statistically if the two populations are different.
