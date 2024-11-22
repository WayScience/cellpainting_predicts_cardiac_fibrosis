# Train logistic regression binary classifers

In this submodule, we split the `Plate 4` data into training, testing, and holdout datasets and train binary logistic regression models to classify single-cells as either coming from healthy/nonfailing or failing hearts.

![plate_4_platemap](../../metadata/platemap_figures/localhost231120090001_platemap_figure.png)

We train `6` total models:

## 1. All features model

We train a final and shuffled model using all the 625 most significant features.
More information can be found in the [main README](../README.md) for this module, where it is explained how the models are trained.

## 2. F-actin only model

We train a final and shuffled model using only the F-actin features from the original 625 features.
We do not include any Correlation feature with `actin` included since these contain another channel which means it does not fit in either feature group.

## 3. Rest/without F-actin model

We train a final and shuffled model using the `rest` of the features that do not include the F-actin features.

**We perform evaluations on these models to assess performance in the next module: [1.evaluate_models](../1.evaluate_models/)**
