"""
This collection of functions performs signed Earth Mover's Distance (EMD) calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Tuple, Dict

def compute_null_emd_range(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    reference_query: str,
    num_permutations: int = 1000,
    random_seed: int = 0,
) -> Tuple[float, float]:
    """
    Calculate the expected range of signed EMDs under the null hypothesis
    that the two populations are not different by independently shuffling 
    each feature column.

    Args:
        reference_df (pd.DataFrame): DataFrame for the reference group.
        comparison_df (pd.DataFrame): DataFrame for the comparison group.
        reference_query (str): Query string to select the reference group from the combined DataFrame.
            Example: 'Metadata_cell_type == "healthy" and Metadata_treatment == "DMSO"'
        num_permutations (int): Number of shuffles to perform.
        random_seed (int): Seed for reproducibility.

    Returns:
        (float, float): 5th and 95th percentiles of signed EMDs across all permutations and features.
    """
    combined_df = pd.concat([reference_df, comparison_df], ignore_index=True)
    feature_cols = [col for col in combined_df.columns if not col.startswith("Metadata_")]

    all_signed_emds = []

    for i in range(num_permutations):
        shuffled_df = combined_df.copy()
        rng = np.random.default_rng(seed=random_seed + i)
        for col in feature_cols:
            shuffled_df[col] = rng.permutation(shuffled_df[col].values)

        # Split into two groups based on the reference query
        group1 = shuffled_df.query(reference_query)
        group2 = shuffled_df[~shuffled_df.index.isin(group1.index)]

        for col in feature_cols:
            vals1 = group1[col].dropna()
            vals2 = group2[col].dropna()
            if len(vals1) == 0 or len(vals2) == 0:
                continue
            emd = wasserstein_distance(vals1, vals2)
            sign = np.sign(np.mean(vals2) - np.mean(vals1))
            all_signed_emds.append(sign * emd)

    lower = np.percentile(all_signed_emds, 5)
    upper = np.percentile(all_signed_emds, 95)
    return lower, upper


def compute_signed_emd_per_feature(
    reference_df: pd.DataFrame, comparison_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute the signed Earth Mover's Distance (EMD) for each feature between two DataFrames.

    Args:
        reference_df (pd.DataFrame): The pandas DataFrame containing the "reference" data or
            what is being used as the base to compare to.
        comparison_df (pd.DataFrame): The pandas DataFrame containing the "comparison" data or
            what is being compared against the reference.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the signed EMD for each feature.
    """
    # Filter to only feature columns (non-Metadata)
    reference_features = reference_df.loc[
        :, ~reference_df.columns.str.startswith("Metadata_")
    ]
    comparison_features = comparison_df.loc[
        :, ~comparison_df.columns.str.startswith("Metadata_")
    ]

    # Only process features shared by both
    shared_features = reference_features.columns.intersection(
        comparison_features.columns
    )
    if shared_features.empty:
        raise ValueError(
            "No shared features between reference and comparison DataFrames."
        )
    not_shared = set(reference_features.columns).symmetric_difference(
        comparison_features.columns
    )
    if not_shared:
        print(f"Features not shared between reference and comparison: {not_shared}")

    # Instantiate results list
    results = []

    # Compute signed EMD for each shared feature
    for feature in shared_features:
        ref_values = reference_features[feature].dropna()
        comp_values = comparison_features[feature].dropna()

        if len(ref_values) == 0 or len(comp_values) == 0:
            continue  # skip if either side is empty

        emd = wasserstein_distance(ref_values, comp_values)
        # Determine the direction of the EMD based on the median values (com > ref = positive EMD, com < ref = negative EMD)
        direction = np.sign(np.median(comp_values) - np.median(ref_values))
        signed_emd = emd * direction

        # Append the result for this feature
        results.append({"feature": feature, "signed_emd": signed_emd})

    # Convert results to DataFrame with all features and scores
    return pd.DataFrame(results)
