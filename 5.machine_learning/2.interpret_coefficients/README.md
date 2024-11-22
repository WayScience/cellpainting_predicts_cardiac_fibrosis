# Interpret coefficients from the all features model

In this submodule, we are only extracting and visualizing the coefficients from the all features model.

We extract the coefficients for each of the 625 features.
We visualize the coefficients using two plots.

1. Heatmap plot looking at the top coefficient across feature measurements and organelles
2. Ranked scatterplot showing the distribution of the coefficients starting from the top (most positive; predicts healthy cells) to the bottom (most negative; predicts failing cells)

## Single-cell crops

We visualize representative single-cells from each of the top features per organelle.
We find the top features for predicting the failing and healthy cells (12 total single-cells).

Single-cell crops are extracted and saved.
We use Fiji to further enhance the crops by being able to add color, create composites, and add a scale bar (1 pixel/uM).
