import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_dataframe(df, drop_na=False, reset_index=True):
    """
    Perform basic cleaning operations on a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        drop_na (bool): Whether to drop rows with missing values.
        reset_index (bool): Whether to reset the index after cleaning.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        return df

    if drop_na:
        df = df.dropna()

    if reset_index:
        df = df.reset_index(drop=True)

    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"

    if df.empty:
        return False, "DataFrame is empty"

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

    return True, "DataFrame is valid"