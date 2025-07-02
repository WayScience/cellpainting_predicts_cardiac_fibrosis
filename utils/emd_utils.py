"""
This collection of functions performs signed Earth Mover's Distance (EMD) calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

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
