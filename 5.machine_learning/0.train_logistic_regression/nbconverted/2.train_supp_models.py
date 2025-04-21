#!/usr/bin/env python
# coding: utf-8

# ## Train machine learning models where we split the features by organelle
# 
# In this notebook, we will split the 625 features used in the overall final and shuffled models into two groups:
# 
# 1. Only F-actin features
# 2. The rest of the features (e.g., nucleus, ER, Golgi/plasma membrane, and Mitochondria)
# 
# NOTE: Prior to splitting based on `Actin` in the feature name versus not, we decided to remove any `correlation` features that include `Actin` because a correlation feature looks at two channels so it would not fit it actin only or rest.
# Once these features are removed, we then split the features.
# 
# We will train a final and shuffled model for each of these feature sets to address the comments from manuscript review.

# ## Import libraries

# In[1]:


import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import parallel_backend

sys.path.append("../../utils")
from training_utils import downsample_data, get_X_y_data


# ## Set paths and variables

# In[2]:


# set numpy seed to make sure any random operations performs are reproducible
np.random.seed(0)

# path to training data set
training_data_path = pathlib.Path("./data/training_data.csv")

# Metadata column used for prediction class
label = "Metadata_cell_type"

# Directory for models to be outputted
model_dir = pathlib.Path("./models/supp_models")
model_dir.mkdir(exist_ok=True, parents=True)

# Directory for label encoder
encoder_dir = pathlib.Path("./encoder_results")
encoder_dir.mkdir(exist_ok=True, parents=True)


# ## Load in training data

# In[3]:


df = pd.read_csv(training_data_path)

print(df.shape)
df.head()


# ## Perform downsampling on training data and output as data frame

# In[4]:


# load in training plate 4 data as downsampled to lowest class
downsample_df = downsample_data(path_to_data=training_data_path, label=label)

print(downsample_df.shape)
print(downsample_df["Metadata_cell_type"].value_counts())
downsample_df.head()


# ## Split the downsample data into two dataframes that contains either only actin features or the rest

# In[5]:


# Separate metadata columns
metadata_columns = [col for col in downsample_df.columns if col.startswith("Metadata_")]

# Separate features columns (filtering out features containing both "Correlation" and "Actin")
filtered_columns = [
    col
    for col in downsample_df.columns
    if not col.startswith("Metadata_") and not ("Actin" in col and "Correlation" in col)
]

# Now split into Actin and non-Actin features
actin_features_columns = [col for col in filtered_columns if "Actin" in col]
rest_features_columns = [col for col in filtered_columns if "Actin" not in col]

# Create the new DataFrames
actin_features_df = downsample_df[metadata_columns + actin_features_columns]
rest_features_df = downsample_df[metadata_columns + rest_features_columns]

# Print the lengths of the new DataFrames
print("Number of feature columns in actin_features_df:", len(actin_features_columns))
print("Number of feature columns in rest_features_df:", len(rest_features_columns))


# ## Get X and y data for both final and shuffled models

# In[6]:


# Store the DataFrames for features
feature_dfs = {"actin": actin_features_df, "rest": rest_features_df}

# Dictionary to store training and shuffled training data
training_data = {}

# Loop through each feature DataFrame (actin and rest)
for feature_type, df in feature_dfs.items():
    # Get non-shuffled training data
    X_train, y_train = get_X_y_data(df=df, label=label, shuffle=False)
    # Get shuffled training data
    X_shuffled_train, y_shuffled_train = get_X_y_data(df=df, label=label, shuffle=True)

    # Store both in the dictionary with keys for each type of data (actin, rest)
    training_data[feature_type] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_shuffled_train": X_shuffled_train,
        "y_shuffled_train": y_shuffled_train,
    }


# ## Encode labels in both shuffled and non-shuffled
# 
# **Note:** Failing will be considered as 0 and Healthy will be 1.

# In[7]:


# Initialize LabelEncoder
le = LabelEncoder()

# Loop through each feature DataFrame (actin and rest)
for feature_type, df in feature_dfs.items():
    # Get non-shuffled training data
    X_train, y_train = get_X_y_data(df=df, label=label, shuffle=False)

    # Fit the LabelEncoder on the non-shuffled labels
    le.fit(y_train)

    # Encode the labels for both non-shuffled and shuffled data
    y_train_encoded = le.transform(y_train)
    X_shuffled_train, y_shuffled_train = get_X_y_data(df=df, label=label, shuffle=True)
    y_shuffled_train_encoded = le.transform(y_shuffled_train)

    # Store the encoded labels and feature data in the dictionary
    training_data[feature_type] = {
        "X_train": X_train,
        "y_train": y_train_encoded,  # Use the encoded labels
        "X_shuffled_train": X_shuffled_train,
        "y_shuffled_train": y_shuffled_train_encoded,  # Use the encoded labels
    }

# Print the class mapping to see the encoding
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Class Mapping:")
print(class_mapping)


# ## Train the models
# 
# **Note:** We will be using RandomizedSearchCV to hyperparameterize the model since we have a larger dataset and it will be easier to try random combinations than all combinations.

# ### Set up the model and hyper parameter method

# In[8]:


# Set folds for k-fold cross validation (default is 5)
straified_k_folds = StratifiedKFold(n_splits=10, shuffle=False)

# Set Logistic Regression model parameters (use default for max_iter)
logreg_params = {
    "penalty": "elasticnet",
    "solver": "saga",
    "max_iter": 1000,
    "n_jobs": -1,
    "random_state": 0,
    "class_weight": "balanced",
}

# Define the hyperparameter search space for RandomizedSearchCV
param_dist = {
    "C": np.logspace(-3, 3, 7),
    "l1_ratio": np.linspace(0, 1, 11),
}

# Set the random search hyperparameterization method parameters (used default for "cv" and "n_iter" parameter)
random_search_params = {
    "param_distributions": param_dist,
    "scoring": "f1_weighted",
    "random_state": 0,
    "n_jobs": -1,
    "cv": straified_k_folds,
}


# ### Train final and shuffled models

# In[9]:


# Initialize Logistic Regression and RandomizedSearchCV
final_logreg = LogisticRegression(**logreg_params)
final_random_search = RandomizedSearchCV(final_logreg, **random_search_params)

# Loop through the training data dictionary for both non-shuffled and shuffled data
for feature_type, data in training_data.items():
    # Get the non-shuffled and shuffled data for the current feature type
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_shuffled_train = data["X_shuffled_train"]
    y_shuffled_train = data["y_shuffled_train"]

    # Prevent the convergence warning in sklearn, it does not impact the result
    with parallel_backend("multiprocessing"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )

            # Train the model for non-shuffled training data
            print(f"Training final model for {feature_type} features (non-shuffled)...")
            final_random_search.fit(X_train, y_train)
            print(
                f"Optimal parameters for {feature_type} features (non-shuffled):",
                final_random_search.best_params_,
            )

            # Save the final model for non-shuffled data using joblib
            final_model_filename = model_dir / f"{feature_type}_final_downsample.joblib"
            dump(final_random_search.best_estimator_, final_model_filename)
            print(f"Model saved as: {final_model_filename}")

            # Train the model for shuffled training data
            print(f"Training final model for {feature_type} features (shuffled)...")
            final_random_search.fit(X_shuffled_train, y_shuffled_train)
            print(
                f"Optimal parameters for {feature_type} features (shuffled):",
                final_random_search.best_params_,
            )

            # Save the final model for shuffled data using joblib
            shuffled_final_model_filename = (
                model_dir / f"{feature_type}_shuffled_downsample.joblib"
            )
            dump(final_random_search.best_estimator_, shuffled_final_model_filename)
            print(f"Model saved as: {shuffled_final_model_filename}")

