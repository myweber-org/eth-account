import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def impute_missing_with_median(data, column):
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)
    return data

def remove_duplicates(data, subset=None):
    if subset:
        data_cleaned = data.drop_duplicates(subset=subset)
    else:
        data_cleaned = data.drop_duplicates()
    return data_cleaned

def standardize_column(data, column):
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std
    return data

def clean_dataset(data, numeric_columns, categorical_columns=None):
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if cleaned_data[col].isnull().any():
            cleaned_data = impute_missing_with_median(cleaned_data, col)
        outliers = detect_outliers_iqr(cleaned_data, col)
        if not outliers.empty:
            median_val = cleaned_data[col].median()
            cleaned_data.loc[outliers.index, col] = median_val
        cleaned_data = standardize_column(cleaned_data, col)
    
    if categorical_columns:
        for col in categorical_columns:
            if cleaned_data[col].isnull().any():
                mode_val = cleaned_data[col].mode()[0]
                cleaned_data[col].fillna(mode_val, inplace=True)
    
    cleaned_data = remove_duplicates(cleaned_data)
    return cleaned_data

def validate_data(data, required_columns):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def get_data_summary(data):
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns)
    }
    return summary
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {col}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {
        'feature1': [10, 12, 14, 100, 15, 13, 11, 9, 16, 12],
        'feature2': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    
    numeric_cols = ['feature1', 'feature2']
    cleaned_df = clean_dataset(df, numeric_cols)
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    main()