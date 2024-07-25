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
    data_set: str,
) -> pd.DataFrame:
    """Generate a data frame with the info for a confusion matrix

    Args:
        model_path (pathlib.Path): path to the model to load in and apply to dataset (either "final" or "shuffled")
        data_dir (pathlib.Path): path to directory with the datasets to evaluate
        encoder_path (pathlib.Path): path to encoder output to use for applying class to label
        label (str): name of the metadata column used for classification to load in the data
        data_set (str): name of the data set you want to find confusion matrix data for 

    Returns:
        pd.DataFrame: data frame containing the confusion matrix data for a given data set
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

    # create confusion matrix
    conf_mat = confusion_matrix(y_binary, y_predict, labels=model.classes_)
    conf_mat = pd.DataFrame(conf_mat, columns=model.classes_, index=model.classes_)

    # use stack to restructure dataframe into tidy long format
    conf_mat = conf_mat.stack()
    # reset index must be used to make indexes at level 0 and 1 into individual columns
    # these columns correspond to true label and predicted label, and are set as indexes after using stack()
    conf_mat = pd.DataFrame(conf_mat).reset_index(level=[0, 1])
    conf_mat.columns = ["True_Label", "Predicted_Label", "Count"]

    # calculate recall for each class
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
