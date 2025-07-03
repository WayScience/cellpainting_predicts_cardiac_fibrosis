"""
This collection of functions performs signed Earth Mover's Distance (EMD) calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

def compute_median_baseline_emd(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    num_permutations: int = 100,
    random_seed: int = 0,
) -> float:
    """Calculate the baseline median Earth Mover's Distance (EMD) between two dataframes using permutation testing.
    This represents the value at which we would expect no change in the features between the two groups.
    EMD is not signed, so we get one value that we will then add sign to later as a range.

    Args:
        reference_df (pd.DataFrame): The pandas DataFrame containing the "reference" data or
            what is being used as the base to compare to.
        comparison_df (pd.DataFrame): The pandas DataFrame containing the "comparison" data or
            what is being compared against the reference.
        num_permutations (int, optional): Number of permutations of the data that will be performed.
            Defaults to 100.

    Returns:
        float: Median EMD value representing the baseline distance for no change between the two groups.
    """
    # Filter out metadata columns (won't be used in EMD calculation)
    df1_features = reference_df.loc[
        :, ~reference_df.columns.str.startswith("Metadata_")
    ]
    df2_features = comparison_df.loc[
        :, ~comparison_df.columns.str.startswith("Metadata_")
    ]

    # Get the shared features between the two dataframes (sanity check)
    shared_features = df1_features.columns.intersection(df2_features.columns)

    # Combine the two dataframes for permutation testing
    # This allows us to shuffle the data while keeping the same features
    combined_df = pd.concat(
        [df1_features[shared_features], df2_features[shared_features]],
        ignore_index=True,
    )
    # Collect the number of rows from the reference dataframe (df1)
    n1 = len(df1_features)

    # Instantiate a list to hold the EMD values for each permutation (for median calculation)
    emd_per_permutation = []

    # Perform the permutation test
    for i in range(num_permutations):
        # Shuffle the combined dataframe for entire rows not per column (maintains structure and distribution)
        # This simulates the null hypothesis that there is no difference between the two groups
        permuted = combined_df.sample(
            frac=1, replace=False, random_state=random_seed + i
        ).reset_index(drop=True)
        # Split the permuted dataframe back into two groups (same size as original groups)
        group1 = permuted.iloc[:n1]
        group2 = permuted.iloc[n1:]

        # Instantiate a list to hold the EMD values for each feature within each permutation
        emd_per_feature = []
        # Compute EMD for each shared feature
        for col in shared_features:
            vals1 = group1[col].dropna()
            vals2 = group2[col].dropna()
            if len(vals1) == 0 or len(vals2) == 0:
                continue
            emd = wasserstein_distance(vals1, vals2)
            emd_per_feature.append(emd)

        # Calculate the median EMD for this permutation
        emd_per_permutation.append(np.median(emd_per_feature))

    # Return the median EMD across all permutations
    return np.median(emd_per_permutation)

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
