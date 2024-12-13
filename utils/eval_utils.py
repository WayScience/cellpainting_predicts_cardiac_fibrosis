"""
This utility file holds functions for generating data frames for confusion matrices, F1 scoring, and accuracy metric.
"""

import pathlib

import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from typing import Union
from training_utils import load_data, get_X_y_data


def generate_confusion_matrix_df(
    model_path: pathlib.Path,
    encoder_path: pathlib.Path,
    label: str,
    data_df: pd.DataFrame = None,
    data_name: str = None,
    data_dir: pathlib.Path = None,
) -> pd.DataFrame:
    """Generate a data frame with the info for a confusion matrix

    Args:
        model_path (pathlib.Path): path to the model to load in and apply to dataset (either "final" or "shuffled")
        encoder_path (pathlib.Path): path to encoder output to use for applying class to label
        label (str): name of the metadata column used for classification to load in the data
        data_df (pd.DataFrame, optional): preloaded dataset as a pandas DataFrame
        data_name (str, optional): name of the data set you want to find confusion matrix data for
        data_dir (pathlib.Path, optional): path to directory with the datasets to evaluate

    Returns:
        pd.DataFrame: data frame containing the confusion matrix data for a given data set
    """
    # Load model
    model = load(model_path)

    # Load label encoder if encoder_path is provided
    le = load(encoder_path) if encoder_path else None

    # Load data if not already provided
    if data_df is None:
        if data_dir is None or data_name is None:
            raise ValueError(
                "data_dir and data_set must be provided if data is not preloaded."
            )
        data_path = pathlib.Path(f"{data_dir}/{data_name}_data.csv")
        data_df = load_data(path_to_data=data_path, label=label)

    # Extract X and y from data
    X, y = get_X_y_data(data_df, label)

    # Encode y using the label encoder if provided
    y_binary = le.transform(y) if le else y

    # Predict labels using the model
    y_predict = model.predict(X)

    # Create confusion matrix
    conf_mat = confusion_matrix(y_binary, y_predict, labels=model.classes_)
    conf_mat = pd.DataFrame(conf_mat, columns=model.classes_, index=model.classes_)

    # Restructure confusion matrix into tidy long format
    conf_mat = conf_mat.stack().reset_index(level=[0, 1])
    conf_mat.columns = ["True_Label", "Predicted_Label", "Count"]

    # Calculate recall for each class
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
    model: Union[object, pathlib.Path],
    data_set: Union[pd.DataFrame, pathlib.Path],
    encoder: Union[object, pathlib.Path],
    label: str,
) -> pd.DataFrame:
    """Generate a data frame with the info for a F1 score plot.

    Args:
        model (Union[object, pathlib.Path]): preloaded model or path to the model file to load.
        data_set (Union[pd.DataFrame, pathlib.Path]): preloaded dataset or path to the dataset CSV.
        encoder (Union[object, pathlib.Path]): preloaded encoder or path to the encoder file.
        label (str): name of the metadata column used for classification.

    Returns:
        pd.DataFrame: data frame containing the F1 score data for the given dataset.
    """
    # Load the model if a path is provided
    if isinstance(model, pathlib.Path):
        model = load(model)

    # Load the encoder if a path is provided
    if isinstance(encoder, pathlib.Path):
        encoder = load(encoder)

    # Load the dataset if a path is provided
    if isinstance(data_set, pathlib.Path):
        data_set = pd.read_csv(data_set)

    # get X and y data from the provided dataframe
    X, y = get_X_y_data(df=data_set, label=label)

    # Assign y classes to correct binary using label encoder results
    y_binary = encoder.transform(y)

    # Predictions for morphology feature data
    y_predict = model.predict(X)

    # Get F1 score data
    scores = f1_score(
        y_binary, y_predict, average=None, labels=model.classes_, zero_division=0
    )
    weighted_score = f1_score(
        y_binary, y_predict, average="weighted", labels=model.classes_, zero_division=0
    )

    # Create a DataFrame for the F1 scores
    scores = pd.DataFrame(scores).T
    scores.columns = model.classes_
    scores["Weighted"] = weighted_score

    return scores


def generate_accuracy_score_df(
    data_set: pd.DataFrame,
    label: str,
    model: Union[object, pathlib.Path],
    encoder: Union[object, pathlib.Path],
) -> pd.DataFrame:
    """Generate a DataFrame with accuracy score information for a given dataset.

    Args:
        data_set (pd.DataFrame): Dataset for evaluation.
        label (str): Column name used for classification labels.
        model (Union[object, pathlib.Path]): Preloaded model object or path to the model file.
        encoder (Union[object, pathlib.Path]): Preloaded label encoder object or path to the encoder file.

    Returns:
        pd.DataFrame: DataFrame containing the accuracy score.
    """
    # if the path to model provided, load in the model
    if isinstance(model, pathlib.Path):
        model = load(model)

    # if the path to encoder provided, load in encoder
    if isinstance(encoder, pathlib.Path):
        encoder = load(encoder)

    # get X and y data from the provided dataframe
    X, y = get_X_y_data(df=data_set, label=label)

    # convert labels to binary
    y_binary = encoder.transform(y)
    # generate predictions
    y_predict = model.predict(X)

    # calculate accuracy scores
    accuracy = accuracy_score(y_binary, y_predict)

    return pd.DataFrame([accuracy], columns=["Accuracy"])
