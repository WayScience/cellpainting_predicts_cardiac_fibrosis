#!/usr/bin/env python
# coding: utf-8

# # Evaluate model performance with PR curves, confusion matrixes, and F1 scores

# ## Import libraries

# In[1]:


import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.metrics import precision_recall_curve, auc

sys.path.append("../../utils")
from eval_utils import (
    generate_confusion_matrix_df,
    generate_f1_score_df,
    generate_accuracy_score_df,
)
from training_utils import load_data


# ## Set paths to different datasets and models

# In[2]:


# Directory with datasets
data_dir = pathlib.Path("../0.train_logistic_regression/data")

# Directory with models
models_dir = pathlib.Path("../0.train_logistic_regression/models")

# Path to encoder file
encoder_path = pathlib.Path("../0.train_logistic_regression/encoder_results/label_encoder_log_reg_fs_plate_4.joblib")

# Directory for model figures output
fig_dir = pathlib.Path("./figures")
fig_dir.mkdir(exist_ok=True)

# Directory for result outputs (accuracy scores)
results_dir = pathlib.Path("./results")
results_dir.mkdir(exist_ok=True)

# Load in each model individually
final_model = load(
    pathlib.Path(f"{models_dir}/log_reg_fs_plate_4_final_downsample.joblib")
)
shuffled_model = load(
    pathlib.Path(f"{models_dir}/log_reg_fs_plate_4_shuffled_downsample.joblib")
)


# ## Precision-Recall Curves

# ### Output precision and recall data per dataset applied to each model and combine into one df

# In[3]:


# Define the subdirectory for this plot type within fig_dir
pr_curves_dir = fig_dir / "pr_curves"
pr_curves_dir.mkdir(parents=True, exist_ok=True)  # Create subdirectory if it doesn't exist

# Initialize empty lists to store data for each iteration
precision_list = []
recall_list = []
threshold_list = []
model_type_list = []
data_type_list = []

for model_path in models_dir.iterdir():
    if model_path.is_dir() or model_path.suffix != ".joblib":
        continue  # Skip directories or files that are not model files

    print("Evaluating", model_path.stem.split("_")[5], "model...")
    
    for data_path in data_dir.iterdir():
        print("Applying model to", data_path.stem, "...")
        # load in model to apply to datasets
        model = load(model_path)

        # load in label encoder
        le = load(encoder_path)

        # Load in X and y data from dataset
        X, y = load_data(path_to_data=data_path, label="Metadata_cell_type")

        # Assign y classes to correct binary using label encoder results
        y_binary = le.transform(y)

        # predict class probabilities for morphology feature data
        predicted_probs = model.predict_proba(X)

        # Calculate the precision, recall data
        precision, recall, threshold = precision_recall_curve(
            y_binary, predicted_probs[:, -1]
        )
        threshold = np.append(threshold, np.nan)

        # Append data to lists
        precision_list.extend(precision.tolist())
        recall_list.extend(recall.tolist())
        threshold_list.extend(threshold.tolist())
        model_type_list.extend([model_path.stem.split("_")[5]] * len(precision))
        data_type_list.extend([data_path.stem.split("_")[0]] * len(precision))

# Create a DataFrame from the accumulated data
pr_df = pd.DataFrame(
    {
        "Precision": precision_list,
        "Recall": recall_list,
        "Threshold": threshold_list,
        "Model_Type": model_type_list,
        "Data_Type": data_type_list,
    }
)

# Drop any NA data
pr_df.dropna(inplace=True)

# Show output of all data
print(pr_df.shape)
pr_df.head()


# ### Plot precision-recall curves for training and testing data per model type

# In[4]:


# PR curves with only testing and training data
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Combine model and data type as one column for plotting
pr_df["data_split"] = pr_df["Model_Type"] + " (" + pr_df["Data_Type"] + ")"

# Filter data frame to only show training versus testing
filtered_df = pr_df[pr_df["Data_Type"].isin(["training", "testing"])]

sns.lineplot(
    x="Recall",
    y="Precision",
    hue="data_split",
    style="Model_Type",
    dashes={"final": (1, 0), "shuffled": (2, 2)},
    data=filtered_df,
    linewidth=4,  # Adjust the line width as needed
)

# Set legend location
plt.legend(loc='lower right')

plt.ylim(bottom=0.0, top=1.02)
plt.xlabel("Recall", fontsize=24)
plt.ylabel("Precision", fontsize=24)
plt.title("Precision vs. Recall Plate 4 Cell Type Classification", fontsize=18)

# Adjust x-axis ticks font size
plt.xticks(fontsize=20)

# Adjust y-axis ticks font size and labels
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig(f"{pr_curves_dir}/precision_recall_plate4_downsample.pdf", dpi=500, bbox_inches='tight')

# Avoid showing plot in notebook
plt.close()


# ### Plot precision-recall curves for **training data only** per model type

# In[5]:


# PR curves with only testing and training data
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")

# Combine model and data type as one column for plotting
pr_df["data_split"] = pr_df["Model_Type"] + " (" + pr_df["Data_Type"] + ")"

# Filter data frame to only show training versus testing
filtered_df = pr_df[pr_df["Data_Type"].isin(["training"])]

# Define colors for each condition
colors = {"final (training)": "red", "shuffled (training)": "#ff7f0e"}

sns.lineplot(
    x="Recall",
    y="Precision",
    hue="data_split",
    style="Model_Type",
    dashes={"final": (1, 0), "shuffled": (2, 2)},
    palette=colors,
    data=filtered_df,
    linewidth=2.5  # Adjust the line width as needed
)

plt.legend(loc="lower right", fontsize=15)
plt.ylim(bottom=0.0, top=1.02)
plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.title("Precision vs. Recall Plate 4 Training Only", fontsize=18)

# Adjust x-axis ticks font size
plt.xticks(fontsize=14)

# Adjust y-axis ticks font size and labels
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig(f"{pr_curves_dir}/precision_recall_plate4_downsample_only_training.png", dpi=500)

# Avoid showing plot in notebook
plt.close()


# ### Calculate Area Under the Precision Recall Curve (AUPRC) for the training and testing data (final model)

# In[6]:


# Filter the dataframe for the final model using the original pr_df
final_model_df = pr_df[pr_df["Model_Type"] == "final"]

# Calculate AUPRC for training data
training_data = final_model_df[final_model_df["Data_Type"] == "training"]
training_auprc = auc(training_data["Recall"], training_data["Precision"])

# Calculate AUPRC for testing data
testing_data = final_model_df[final_model_df["Data_Type"] == "testing"]
testing_auprc = auc(testing_data["Recall"], testing_data["Precision"])

# Output the results
print(f"AUPRC for Training Data: {training_auprc:.4f}")
print(f"AUPRC for Testing Data: {testing_auprc:.4f}")


# ### Plot precision-recall curves for holdout data per model type

# In[7]:


# Custom colors for the holdout plot to differentiate between the training/testing plot
custom_colors = ["#CC79A7", "#0072B2", "#009E73", "#882255"]  # Pink, Dark Blue, Teal, Burgundy

plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")

# Combine model and data type as one column for plotting
pr_df["data_split"] = pr_df["Model_Type"] + " (" + pr_df["Data_Type"] + ")"

# Filter data frame to only show holdout datasets
filtered_df = pr_df[pr_df["Data_Type"].isin(["holdout1", "holdout2"])]

sns.lineplot(
    x="Recall",
    y="Precision",
    hue="data_split",
    style="Model_Type",
    dashes={"final": (1, 0), "shuffled": (2, 2)},
    data=filtered_df,
    linewidth=2.5,  # Adjust the line width as needed
    palette=custom_colors  # Apply the custom color palette
)

plt.legend(loc="lower right", fontsize=13)
plt.ylim(bottom=0.0, top=1.02)
plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.title("Precision vs. Recall Plate 4 Holdout Datasets", fontsize=18)

# Adjust x-axis ticks font size
plt.xticks(fontsize=14)

# Adjust y-axis ticks font size and labels
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig(f"{pr_curves_dir}/precision_recall_plate4_holdout_data_downsample.png", dpi=500)

# Close the plot to avoid showing it in the notebook
plt.close()


# ## Confusion Matrices

# ### Generate confusion matrix plots for each data set and model type

# In[8]:


# Define the subdirectory for this plot type within fig_dir
con_matrix_dir = fig_dir / "confusion_matrices"
con_matrix_dir.mkdir(parents=True, exist_ok=True)  # Create subdirectory if it doesn't exist

# List of paths that contains each model
path_list = list(models_dir.glob("*"))

# List of data sets to apply models to
data_set_list = ["training", "testing", "holdout1", "holdout2"]

# Metadata column used for classifying
label = "Metadata_cell_type"

# Iterate over each model path
for model_path in path_list:
    if model_path.is_dir() or model_path.suffix != ".joblib":
        continue  # Skip directories or files that are not model files
    
    # Iterate over each dataset
    for data_set in data_set_list:
        print(
            "Generating plot for",
            data_set.capitalize(),
            model_path.stem.split("_")[5].capitalize(),
        )
        # Generate confusion matrix data frame
        df = generate_confusion_matrix_df(
            model_path=model_path,
            data_dir=data_dir,
            encoder_path=encoder_path,
            label=label,
            data_set=data_set,
        )

        # Rename binary labels to failing versus healthy
        df["True_Label"] = df["True_Label"].replace({0: "Failing", 1: "Healthy"})
        df["Predicted_Label"] = df["Predicted_Label"].replace(
            {0: "Failing", 1: "Healthy"}
        )

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.set_style("whitegrid")

        heatmap = sns.heatmap(
            data=df.pivot(
                index="True_Label", columns="Predicted_Label", values="Recall"
            ),
            annot=df.pivot(
                index="True_Label", columns="Predicted_Label", values="Count"
            ),
            fmt=".0f",
            cmap="Reds",
            square=True,
            vmin=0,
            vmax=1,
            xticklabels=["Failing", "Healthy"],
            yticklabels=["Failing", "Healthy"],
            linewidths=0.5,
            annot_kws={"size": 16},  # Adjusted annotation font size
            mask=(
                df.pivot(
                    index="True_Label", columns="Predicted_Label", values="Count"
                ).isna()
            ),
            cbar_kws={"label": "Recall"},
        )

        # Adjust color bar label font size
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_label("Recall", fontsize=14)  # Adjust colorbar label font size
        colorbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick labels font size

        plt.title(
            f"Confusion Matrix for {data_set.capitalize()} Data - Model: {model_path.stem.split('_')[5].capitalize()}",
            fontsize=20,  # Adjusted title font size
        )
        plt.xticks(fontsize=14)  # Adjusted x-axis tick label font size
        plt.yticks(fontsize=14)  # Adjusted y-axis tick label font size
        plt.tight_layout()

        plt.savefig(
            f"{con_matrix_dir}/plate4_confusion_matrix_{data_set}_{model_path.stem.split('_')[5]}_downsample.png",
            dpi=500,
        )
        plt.close()


# ## F1 scores

# ### Concat F1 scores from all datasets and model types

# In[9]:


# Define the subdirectory for this plot type within fig_dir
f1_scores_dir = fig_dir / "f1_scores"
f1_scores_dir.mkdir(parents=True, exist_ok=True)  # Create subdirectory if it doesn't exist

# List of paths that contains each model
path_list = list(models_dir.glob("*"))

# Metadata column used for classifying
label = "Metadata_cell_type"

# list for the f1 score dataframes per model and data split
f1_scores_dfs = []

# Iterate over each model path
for model_path in path_list:
    if model_path.is_dir() or model_path.suffix != ".joblib":
        continue  # Skip directories or files that are not model files
    
    # Iterate over each dataset
    for data_set in data_set_list:
        print(data_set.capitalize(), model_path.stem.split("_")[5].capitalize())
        
        # Generate f1 score data frame
        f1_scores_df = generate_f1_score_df(
            model_path=model_path,
            data_dir=data_dir,
            encoder_path=encoder_path,
            label=label,
            data_set=data_set,
        )

        # Rename binary labels to failing versus healthy
        f1_scores_df = f1_scores_df.rename(columns={0: "Failing", 1: "Healthy"})

        # Add columns for model and data_set
        f1_scores_df["Model"] = model_path.stem.split("_")[5].capitalize()
        f1_scores_df["Data_Set"] = data_set.capitalize()

        # Append the DataFrame to the list
        f1_scores_dfs.append(f1_scores_df)

concat_f1_scores = pd.concat(f1_scores_dfs, ignore_index=True)


# ### Check if output looks correct

# In[10]:


print(concat_f1_scores.shape)
concat_f1_scores.head()


# ### Plot F1 scores for only testing and training data

# In[11]:


# Set palette for the f1 scores plot
palette = sns.color_palette("Paired")[:2]

# Set figure size
plt.figure(figsize=(10, 6))

# Set y-axis limits
plt.ylim(0, 1)

# Filter the data to include only training and testing sets
filtered_data = concat_f1_scores[
    concat_f1_scores["Data_Set"].isin(["Training", "Testing"])
]

ax = sns.barplot(
    x="Model", y="Weighted", hue="Data_Set", data=filtered_data, palette=palette
)

# Add the weighted values above the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

plt.title("Weighted F1 Score by Model Type")
plt.xlabel("Model Type")
plt.ylabel("Weighted F1 Score")
plt.legend(title="Data Set")
plt.tight_layout()

plt.savefig(f"{f1_scores_dir}/f1_scores_plate4_downsample.png", dpi=500)

# Avoid showing plot in the notebook
plt.close()


# ### Plot F1 scores for holdout data

# In[12]:


# Set palette for the f1 scores plot
palette = sns.color_palette("Paired")[2:4]

# Set figure size
plt.figure(figsize=(10, 6))

# Set y-axis limits
plt.ylim(0, 1)

# Filter the data to include only hold-out sets
filtered_data = concat_f1_scores[
    concat_f1_scores["Data_Set"].isin(["Holdout1", "Holdout2"])
]

ax = sns.barplot(
    x="Model", y="Weighted", hue="Data_Set", data=filtered_data, palette=palette
)

# Add the weighted values above the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

plt.title("Weighted F1 Score by Model Type")
plt.xlabel("Model Type")
plt.ylabel("Weighted F1 Score")
plt.legend(title="Data Set")
plt.tight_layout()

plt.savefig(f"{f1_scores_dir}/f1_scores_plate4_holdout_data_downsample.png", dpi=500)

# Avoid showing plot in the notebook
plt.close()


# ## Generate accuracy scores specifically from the whole plate 4 dataset per heart and save CSV

# In[13]:


# List of paths that contains each model
path_list = list(models_dir.glob("*"))

# Metadata column used for classifying
label = "Metadata_cell_type"

# List for all accuracy dataframes for concat
accuracy_dfs = []

# Iterate over each model path
for model_path in path_list:
    if model_path.is_dir() or model_path.suffix != ".joblib":
        continue  # Skip directories or files that are not model files
    # Iterate over each dataset
    for data_set in data_set_list:
        print(data_set.capitalize(), model_path.stem.split("_")[5].capitalize())

        # Load the dataset
        data_path = data_dir / f"{data_set}_data.csv"
        data_df = pd.read_csv(data_path)

        # Group the data by heart number
        grouped_data = data_df.groupby("Metadata_heart_number")

        # Iterate over each group (heart number)
        for heart_number, df_heart in grouped_data:
            # Generate accuracy data frame
            accuracy_df = generate_accuracy_score_df(
                model_path=model_path,
                data_set=df_heart,
                encoder_path=encoder_path,
                label=label,
            )

            # Rename binary labels to failing versus healthy
            accuracy_df = accuracy_df.rename(columns={0: "Failing", 1: "Healthy"})

            # Add columns for model, data_set, and heart number
            accuracy_df["Model"] = model_path.stem.split("_")[5].capitalize()
            accuracy_df["Data_Set"] = data_set.capitalize()
            accuracy_df["Heart_Number"] = heart_number

            # Append the DataFrame to the list
            accuracy_dfs.append(accuracy_df)

# Merge all of the accuracy scores across model type and data splot
concat_accuracy_scores = pd.concat(accuracy_dfs, ignore_index=True)

# Save results for visualization
concat_accuracy_scores.to_csv(f"{results_dir}/accuracy_scores_per_heart.csv", index=False)

