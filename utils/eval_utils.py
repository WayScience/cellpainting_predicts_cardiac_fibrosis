"""
This utility file holds functions for generating data frames for confusion matrices, F1 scoring, and accuracy metric.
"""

import pathlib

import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from training_utils import load_data, get_X_y_data


def generate_confusion_matrix_df(
    model_path: pathlib.Path,
    data_dir: pathlib.Path,
    encoder_path: pathlib.Path,
    label: str,
    data_set: str = None,
    data_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Generate a data frame with the info for a confusion matrix

    Args:
        model_path (pathlib.Path): path to the model to load in and apply to dataset (either "final" or "shuffled")
        data_dir (pathlib.Path): path to directory with the datasets to evaluate
        encoder_path (pathlib.Path): path to encoder output to use for applying class to label
        label (str): name of the metadata column used for classification to load in the data
        data_set (str, optional): name of the data set you want to find confusion matrix data for (if loading from file)
        data_df (pd.DataFrame, optional): preloaded dataframe to use instead of loading from file

    Returns:
        pd.DataFrame: data frame containing the confusion matrix data for a given data set
    """
    # Load model
    model = load(model_path)

    # Load label encoder
    le = load(pathlib.Path(encoder_path))

    # Load data
    if data_df is None:
        if data_set is None:
            raise ValueError("Either 'df' or 'data_set' must be provided.")
        data_path = data_dir / f"{data_set}_data.csv"
        df = load_data(path_to_data=data_path, label=label)

    # Ensure dataframe contains the required label column
    if label not in df.columns:
        raise ValueError(f"Column '{label}' not found in the provided dataframe.")

    # Extract features and labels
    X, y = df.drop(columns=[label]), df[label]

    # Encode labels
    y_binary = le.transform(y)

    # Model predictions
    y_predict = model.predict(X)

    # Create confusion matrix
    conf_mat = confusion_matrix(y_binary, y_predict, labels=model.classes_)
    conf_mat = pd.DataFrame(conf_mat, columns=model.classes_, index=model.classes_)

    # Reshape to long format
    conf_mat = conf_mat.stack().reset_index()
    conf_mat.columns = ["True_Label", "Predicted_Label", "Count"]

    # Calculate recall
    conf_mat["Recall"] = conf_mat.apply(
        lambda row: (
            row["Count"]
            / conf_mat[conf_mat["True_Label"] == row["True_Label"]]["Count"].sum()
            if conf_mat[conf_mat["True_Label"] == row["True_Label"]]["Count"].sum() != 0
            else 0
        ),
        axis=1,
    )

    return conf_mat


def generate_f1_score_df(
    model_path: pathlib.Path,
    data_dir: pathlib.Path,
    encoder_path: pathlib.Path,
    label: str,
    data_set: str,
) -> pd.DataFrame:
    """Generate a data frame with the info for a F1 score plot

    Args:
        model_path (pathlib.Path): path to the model to load in and apply to dataset (either "final" or "shuffled")
        data_dir (pathlib.Path): path to directory with dataset to evaluate
        encoder_path (pathlib.Path): path to encoder output to use for applying class to label
        label (str): name of the metadata column used for classification to load in the data
        data_set (str): name of the data set you want to find f1 score data for

    Returns:
        pd.DataFrame: data frame containing the f1 score data for a given data set
    """
    # load in model to apply to data sets
    model = load(model_path)

    # load in label encoder
    le = load(pathlib.Path(encoder_path))

    # set path to specific data set
    data_path = pathlib.Path(f"{data_dir}/{data_set}_data.csv")

    # load in X and y data from dataset
    X, y = load_data(path_to_data=data_path, label=label)

    # Assign y classes to correct binary using label encoder results
    y_binary = le.transform(y)

    # predictions for morphology feature data
    y_predict = model.predict(X)

    # Get F1 score data
    scores = f1_score(
        y_binary, y_predict, average=None, labels=model.classes_, zero_division=0
    )
    weighted_score = f1_score(
        y_binary, y_predict, average="weighted", labels=model.classes_, zero_division=0
    )
    scores = pd.DataFrame(scores).T
    scores.columns = model.classes_
    scores["Weighted"] = weighted_score

    return scores


def generate_accuracy_score_df(
    model_path: pathlib.Path,
    data_set: pd.DataFrame,
    encoder_path: pathlib.Path,
    label: str,
) -> pd.DataFrame:
    """Generate a data frame with the info for an accuracy score plot. Requires a loaded in data frame as input

    Args:
        model_path (pathlib.Path): path to the model to load in and apply to dataset (either "final" or "shuffled")
        data_set (pd.DataFrame): pandas data frame of the data to evaluate
        encoder_path (pathlib.Path): path to encoder output to use for applying class to label
        label (str): name of the metadata column used for classification to load in the data

    Returns:
        pd.DataFrame: data frame containing the accuracy data for a given data set
    """
    # load in model to apply to data sets
    model = load(model_path)

    # load in label encoder
    le = load(pathlib.Path(encoder_path))

    # load in X and y data from dataset
    X, y = get_X_y_data(df=data_set, label=label)

    # Assign y classes to correct binary using label encoder results
    y_binary = le.transform(y)

    # predictions for morphology feature data
    y_predict = model.predict(X)

    # Get accuracy score data
    accuracy = accuracy_score(y_binary, y_predict)

    scores = pd.DataFrame([accuracy], columns=["Accuracy"])

    return scores
