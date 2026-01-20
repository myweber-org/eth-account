import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed missing value percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    missing_percentage = df.isnull().mean(axis=1)
    return df[missing_percentage <= threshold].reset_index(drop=True)

def replace_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Replace outliers with column boundaries using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers replaced
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5, standardize=True):
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_threshold (float): Threshold for removing rows with missing values
        outlier_multiplier (float): Multiplier for IQR outlier detection
        standardize (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Step 1: Handle missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After missing value removal: {df_clean.shape}")
    
    # Step 2: Handle outliers
    df_clean = replace_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print("Outliers replaced using IQR method")
    
    # Step 3: Standardize if requested
    if standardize:
        df_clean = standardize_columns(df_clean)
        print("Numeric columns standardized")
    
    return df_cleanimport pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        missing_after = df.isnull().sum().sum()
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Fixed {missing_before - missing_after} missing values")
        print(f"  - Cleaned data saved to: {output_path}")
        print(f"  - Final dataset shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        return False
    
    # Check for required columns if any
    required_columns = []
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    # Create sample CSV
    temp_df = pd.DataFrame(sample_data)
    temp_df.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample_data.csv')
    
    if cleaned_df is not None:
        print("\nFirst few rows of cleaned data:")
        print(cleaned_df.head())
        
        # Clean up sample files
        import os
        os.remove('sample_data.csv')