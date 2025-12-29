
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Columns to check for duplicates. 
            If None, checks all columns.
        fill_missing (bool): If True, fill missing values with column mean for numeric
            columns and mode for categorical columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values if requested
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                # Fill numeric columns with mean
                mean_value = cleaned_df[column].mean()
                cleaned_df[column] = cleaned_df[column].fillna(mean_value)
            else:
                # Fill categorical columns with mode
                mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                cleaned_df[column] = cleaned_df[column].fillna(mode_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        for column in df.columns:
            if df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].mean()
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].median()
                elif fill_missing == 'mode':
                    fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
                else:
                    fill_value = 0 if pd.api.types.is_numeric_dtype(df[column]) else 'Unknown'
                
                df[column] = df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_value}")
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the dataset structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' should be numeric but contains non-numeric data")
    
    return True

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save the cleaned DataFrame to a file.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'age': [25, 30, 30, None, 35],
        'score': [85.5, 92.0, 92.0, 78.5, None]
    })
    
    print("Original dataset:")
    print(sample_data)
    print("\nCleaning dataset...")
    
    cleaned_data = clean_dataset(sample_data, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_data)
    
    # Validate the cleaned data
    try:
        validate_data(cleaned_data, 
                     required_columns=['id', 'name', 'age', 'score'],
                     numeric_columns=['age', 'score'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")