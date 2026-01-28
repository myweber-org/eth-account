
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    # For numerical columns, fill with median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataset(df):
    """
    Validate the dataset for common issues.
    """
    validation_report = {}
    
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    validation_report['missing_values'] = missing_values
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    validation_report['duplicates'] = duplicates
    
    # Check data types
    validation_report['dtypes'] = df.dtypes.to_dict()
    
    # Check shape
    validation_report['shape'] = df.shape
    
    return validation_report

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 40, 40, 45],
        'score': [85.5, 90.0, 78.5, None, 92.5, 92.5, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    # Clean the dataset
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df)
    print("\nValidation report:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    outlier_report = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df, outliers_removed = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            outlier_report[col] = outliers_removed
            
            # Standardize the column
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
            
            # Normalize the column
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df, outlier_report

def validate_data(df, required_columns, numeric_columns=None):
    """
    Validate dataset structure and content.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_report = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if numeric_columns:
        numeric_stats = {}
        for col in numeric_columns:
            if col in df.columns:
                numeric_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': stats.skew(df[col].dropna())
                }
        validation_report['numeric_stats'] = numeric_stats
    
    return validation_report

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(50, 15, 100),
        'feature_b': np.random.exponential(2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 200
    sample_data.loc[20, 'feature_b'] = 50
    
    print("Original data shape:", sample_data.shape)
    
    # Validate data
    validation = validate_data(sample_data, 
                              required_columns=['id', 'feature_a', 'feature_b'],
                              numeric_columns=['feature_a', 'feature_b'])
    print("\nValidation Report:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    # Clean data
    cleaned_data, outliers = clean_dataset(sample_data, 
                                          numeric_columns=['feature_a', 'feature_b'])
    
    print(f"\nOutliers removed: {outliers}")
    print("Cleaned data shape:", cleaned_data.shape)
    print("\nCleaned data columns:", cleaned_data.columns.tolist())import pandas as pd
import numpy as np
import logging

def clean_csv_data(file_path, output_path=None, drop_na=True, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path for cleaned CSV output. Defaults to None.
        drop_na (bool): Whether to drop rows with missing values. Defaults to True.
        fill_strategy (str): Strategy for filling missing values if drop_na is False.
                            Options: 'mean', 'median', 'mode', 'zero'. Defaults to 'mean'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        original_shape = df.shape
        
        logging.info(f"Loaded data from {file_path}. Shape: {original_shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        duplicates_removed = original_shape[0] - df.shape[0]
        logging.info(f"Removed {duplicates_removed} duplicate rows.")
        
        # Handle missing values
        if drop_na:
            df = df.dropna()
            logging.info("Dropped rows with missing values.")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_strategy == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # For non-numeric columns, fill with empty string
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            df[non_numeric_cols] = df[non_numeric_cols].fillna('')
            
            logging.info(f"Filled missing values using {fill_strategy} strategy.")
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        final_shape = df.shape
        logging.info(f"Data cleaning complete. Final shape: {final_shape}")
        logging.info(f"Rows removed: {original_shape[0] - final_shape[0]}")
        logging.info(f"Columns: {original_shape[1]} (unchanged)")
        
        # Save cleaned data if output path is provided
        if output_path:
            df.to_csv(output_path, index=False)
            logging.info(f"Cleaned data saved to {output_path}")
        
        return df
        
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"File is empty: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['warnings'].append(f"Column '{col}' contains infinite values")
    
    # Add summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return validation_results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    try:
        cleaned_df = clean_csv_data(
            file_path='input_data.csv',
            output_path='cleaned_data.csv',
            drop_na=False,
            fill_strategy='median'
        )
        
        validation = validate_dataframe(cleaned_df)
        print(f"Validation results: {validation}")
        
    except Exception as e:
        print(f"Error: {e}")