
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data_minmax(dataframe, columns):
    normalized_df = dataframe.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(input_path, output_path, outlier_cols=None, normalize_cols=None):
    try:
        df = pd.read_csv(input_path)
        
        if outlier_cols:
            df = remove_outliers_iqr(df, outlier_cols)
        
        if normalize_cols:
            df = normalize_data_minmax(df, normalize_cols)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    outlier_columns = ['age', 'income', 'score']
    normalize_columns = ['income', 'score']
    
    clean_dataset(input_file, output_file, outlier_columns, normalize_columns)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def main():
    # Example usage
    import pandas as pd
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'value': np.random.normal(100, 15, 1000)
    })
    
    print(f"Original data shape: {sample_data.shape}")
    cleaned_data = remove_outliers_iqr(sample_data, 'value')
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Number of outliers removed: {len(sample_data) - len(cleaned_data)}")

if __name__ == "__main__":
    main()