import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from typing import Union


# Define function to preform bootstrapping
def bootstrap_roc_auc(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    n_bootstraps: int = 1000,
) -> np.ndarray:
    """
    Perform bootstrapping to compute the distribution of ROC AUC scores.

    This function generates a bootstrapped distribution of ROC AUC scores by
    resampling the provided true labels and predicted probabilities with
    replacement.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1) for the dataset.

    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or scores for the positive class.

    n_bootstraps : int, optional, default=1000
        Number of bootstrap iterations to perform.

    Returns:
    -------
    bootstrapped_scores : np.ndarray
        An array of bootstrapped ROC AUC scores. Each element represents the
        ROC AUC computed for a resampled dataset.
    """
    # list for the scores to be appended to
    bootstrapped_scores = []

    # loop through and create the amount of bootstrap samples and calculate ROC scores
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        # evaluate if the subsample has both classes
        if len(np.unique(y_true[indices])) < 2:
            # skip this subsample if it doesn't have both classes
            continue
        # if there are both classes, then calculate the score
        else:
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)

    return np.array(bootstrapped_scores)
