
import pandas as pd

def clean_dataframe(df, column_name, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric column threshold.
    Removes rows where the column value is NaN.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    df_clean = df.dropna(subset=[column_name])

    if keep_above:
        filtered_df = df_clean[df_clean[column_name] > threshold]
    else:
        filtered_df = df_clean[df_clean[column_name] <= threshold]

    return filtered_df.reset_index(drop=True)