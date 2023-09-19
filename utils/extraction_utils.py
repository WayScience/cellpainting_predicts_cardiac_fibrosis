"""
This file holds functions to extract image features from the outputted sqlite file from CellProfiler. These functions are based on the functions from the
cells.SingleCells class in Pycytominer. This file also hold a function to add single cell counts
to the single cell count dataframes.
"""

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import pathlib


def load_sqlite_as_df(
    sqlite_file: str,
    image_table_name: str = "Per_Image",
) -> pd.DataFrame:
    """
    load in table with image feature data from sqlite file

    Parameters
    ----------
    sqlite_file : str
        string of path to the sqlite file
    image_table_name : str
        string of the name with the image feature data (default = "Per_Image")

    Returns
    -------
    pd.DataFrame:
        dataframe containing image feature data
    """
    # connect to the sqlite file to be able read the contents
    engine = create_engine(sqlite_file)
    conn = engine.connect()

    # read and output all columns and rows from table into a dataframe to
    image_query = f"select * from {image_table_name}"
    image_df = pd.read_sql(sql=image_query, con=conn)

    return image_df


def extract_image_features(image_feature_categories, image_df, image_cols, strata):
    """Extract image features based on set image categories.
    This is pulled from Pycytominer cyto_utils util.py and editted.

    Parameters
    ----------
    image_feature_categories : list of str
        Input image feature groups to extract from the image table including the prefix (e.g. ["Image_Correlation", "Image_ImageQuality"])
    image_df : pandas.core.frame.DataFrame
        Image dataframe.
    image_cols : list of str
        Columns to select from the image table.
    strata :  list of str
        The columns to groupby and aggregate single cells.
    Returns
    -------
    image_features_df : pandas.core.frame.DataFrame
        Dataframe with extracted image features.
    """
    # Extract Image features from image_feature_categories
    image_features = list(
        image_df.columns[
            image_df.columns.str.startswith(tuple(image_feature_categories))
        ]
    )

    # Add image features to the image_df
    image_features_df = image_df[image_features]

    # Add image_cols and strata to the dataframe
    image_features_df = pd.concat(
        [image_df[list(np.union1d(image_cols, strata))], image_features_df], axis=1
    )

    return image_features_df


def add_sc_count_metadata(data_path: pathlib.Path):
    """
    This function loads in the saved csv from Pycytominer (e.g. normalized, etc.), adds the single cell counts for
    each well as metadata, and saves the csv to the same place (as a csv.gz file)

    Parameters
    ----------
    data_path : pathlib.Path
        path to the csv.gz files outputted from Pycytominer (this is the same path as the output path)
    """
    data_df = pd.read_csv(data_path, compression="gzip")

    # this creates a dataframe with the number of single cells calculated using the groupby function
    merged_data = (
        data_df.groupby(["Metadata_Well"])["Metadata_Well"]
        .count()
        .reset_index(name="Metadata_number_of_singlecells")
    )

    # the dataframe with the single cell count is merged into the extracted features dataframe (number of single cell column added to the end)
    data_df = data_df.merge(merged_data, on="Metadata_Well")
    # pop out the column from the dataframe
    singlecell_column = data_df.pop("Metadata_number_of_singlecells")
    # insert the column as the second index column in the dataframe
    data_df.insert(2, "Metadata_number_of_singlecells", singlecell_column)

    # saves dataframe as csv to the same path
    data_df.to_csv(data_path)
