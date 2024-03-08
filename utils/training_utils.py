"""
This utility file contains the function to load in training data as final or shuffled to be used in training a machine learning model.
"""

import numpy as np
import pandas as pd
import pathlib
from typing import Tuple


# set numpy seed to make random operations (shuffling data) reproducible
np.random.seed(0)


def get_X_y_data(
    df: pd.DataFrame, label: str, shuffle: bool = False
) -> Tuple[np.array, np.array]:
    """Get X (feature space) and labels (predicting class) from pandas Data frame

    Args:
        df (pd.DataFrame): Data frame containing morphology.
        label (str): Name of the Metadata column being used as the predicting class
        shuffle (bool, optional): Shuffle the feature columns to get a shuffled dataset. Defaults to False.

    Returns:
        Tuple[np.array, np.array]: Returns np.arrays for the feature space (X) and the predicting class (y)
    """
    # Remove "Metadata" columns from df, leaving only the feature space
    X = df.loc[:, ~df.columns.str.contains("Metadata")].values

    # Extract class label
    y = df.loc[:, [label]].values
    # Make labels as array for use in machine learning
    y = np.ravel(y)

    # If shuffle is True, shuffled the columns independently for the feature space for training a model
    if shuffle == True:
        for column in X.T:
            np.random.shuffle(column)

    return X, y


def load_data(
    path_to_data: pathlib.Path, label: str, shuffle: bool = False
) -> Tuple[np.array, np.array]:
    """Load in data from a path as X (feature space) and labels (predicting class)

    Args:
        path_to_data (pathlib.Path): Path to the CSV contain morphology data that you want to load in. Expected format is CSV file.
        label (str): Name of the Metadata column being used as the predicting class
        shuffle (bool, optional): Shuffle the feature columns to get a shuffled dataset. Defaults to False.

    Returns:
        Tuple[np.array, np.array]: Returns np.arrays for the feature space (X) and the predicting class (y)
    """
    # Load in data frame from CSV, if not CSV file then return error
    if path_to_data.suffix.lower() == ".csv":
        # Load the CSV file
        df = pd.read_csv(path_to_data, index_col=0)
    else:
        print("File does not have a CSV extension. Current expected input is CSV.")

    # Get X, y data from loaded in data frame
    X, y = get_X_y_data(df=df, label=label, shuffle=shuffle)

    return X, y


def downsample_data(path_to_data: pathlib.Path, label: str) -> pd.DataFrame:
    """Load in data from path and down sample to the lowest class, returning a data frame to use for retrieving X, y data

    Args:
        df (pd.DataFrame): Input DataFrame.
        label (str): Name of the Metadata column being used as the predicting class.

    Returns:
        pd.DataFrame: Downsampled DataFrame.
    """
    # Load in data frame from CSV, if not CSV file then return error
    if path_to_data.suffix.lower() == ".csv":
        # Load the CSV file
        df = pd.read_csv(path_to_data, index_col=0)
    else:
        print("File does not have a CSV extension. Current expected input is CSV.")

    # Find class with lowest sample from label
    min_samples = df[label].value_counts().min()

    # Downsample classes to the lowest label
    df_downsampled = df.groupby(label, group_keys=False).apply(
        lambda x: x.sample(min_samples)
    )

    return df_downsampled
