# 5. Machine learning

In this module, we train a logistic regression elasticnet machine learning model to predict if a single cell is either failing or healthy based on the feature selected morphology features from Plate 4. 

The metadata column used for the prediction classes is called `Metadata_cell_type`.
There are different heart numbers for each class, called `Metadata_heart_number`, where there are 2 healthy hearts (`#2` and `#7`) and 4 failing hearts (`#4`, `#19`, `#23`, and `#29`).

## Splitting the data

We split the data into hold-out, training, and testing datasets using the Plate 4 feature selected data.

We first split out the hold-out data, which consisted of two different datasets:

1. **Holdout1**: All wells from DMSO treated healthy heart #7 and all wells from one failing heart (randomly selected, ended up being #29).
2. **Holdout2**: One well from each heart (both failing and healthy) except for the hearts held-out in Holdout1.

After removing hold-out data, we then split the remaining data **70% training** and **30% testing**.

## Training the models

We trained two logistic regression models:

- **Final model**: Model that we expect to perform better than random
- **Shuffled baseline model**: Model using training data where the column data is independently shuffled, where we expect that there will only be random noise.

Prior to training, we downsample the training data so that the single-cell count for the failing class are the same as the healthy class to avoid under-fitting due to lack of representation in one class.

We use the [logistic regression function from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for our models. 
We use the following parameters:

- `penalty = elasticnet` - Use both L1 and L2 penalty terms.
- `solver = saga` - Optimal for multi-class problems and larger datasets.
- `max_iter = 1000` - Maximum number of iterations taken for the solvers to converge, default is 100. We increased it to improve performance.
- `random_state = 0` - Used when using `saga` as the solver which shuffles the data when training.
- `class_weight = balanced` - Though we downsample to balance the classes, we still include the balanced class_weight as a way to confirm the model is balancing the classes.

We use the [randomized search CV function from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to perform optimize the hyper parameters where it maximizes the F1 weighted score. 
There are other metrics to choose for maximizing, but we use F1 weighted score since it is a harmonic mean of precision and recall as defined by the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). 
We found this to be more optimal than just using either precision or recall alone.
We also used the [stratified k-fold cross-validation function from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) to make sure each fold contains equal number of cells from each class and help with avoiding over-fitting.
We increase the number of folds to 10 from the default of 5 to allow for more chance of reducing over-fitting in the model and since we have a fairly large training dataset, we can allow for more splits without having too small of splits.
Below is the hyper parameter search space we used:

- `C` - Inverse of the regularization strength, we set the range of values as `[1e-03, 1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03]`.
- `l1_ratio` - The Elastic-Net mixing parameter, where 0 means using the L2 penalty, 1 means using the L1 penalty, and in between is a mix of the two penalities. We set the range of values as `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.

## Evaluating the model

We evaluate the each model's performance by using three different plots:

1. Precision-recall curves
2. Confusion matrices
3. F1-scores

We create plots for each model type (`final` or `shuffled`) and each data split (`training`, `testing`, `holdout1`, `holdout2`).

We found that down-sampling the classes prior to training the model significantly improved the performance on the training and testing datasets.
There is high performance when applying the model to data it has never seen (e.g., holdout1 and holdout2).
The shuffled model performs poorly compared to the final model, indicating the our model is able to detect a significant pattern between failing and healthy cells.

## Running the notebooks

To perform the data splitting, training, and evaluation, run the below code in terminal:

```bash
cd 5.machine_learning
source machine_learning.sh
```
