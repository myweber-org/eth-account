
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
    print("\nCleaned data columns:", cleaned_data.columns.tolist())