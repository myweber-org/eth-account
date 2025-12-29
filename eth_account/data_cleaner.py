
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column_zscore(dataframe, column):
    """
    Normalize column using z-score normalization
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_col = stats.zscore(dataframe[column])
    result_df = dataframe.copy()
    result_df[f"{column}_normalized"] = normalized_col
    
    return result_df

def min_max_normalize(dataframe, column, new_min=0, new_max=1):
    """
    Apply min-max normalization to specified column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    old_min = dataframe[column].min()
    old_max = dataframe[column].max()
    
    if old_max == old_min:
        normalized_values = np.zeros(len(dataframe))
    else:
        normalized_values = ((dataframe[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    
    result_df = dataframe.copy()
    result_df[f"{column}_scaled"] = normalized_values
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    result_df = dataframe.copy()
    
    if columns is None:
        columns = dataframe.columns
    
    for col in columns:
        if col not in dataframe.columns:
            continue
            
        if dataframe[col].isnull().any():
            if strategy == 'mean':
                fill_value = dataframe[col].mean()
            elif strategy == 'median':
                fill_value = dataframe[col].median()
            elif strategy == 'mode':
                fill_value = dataframe[col].mode()[0]
            elif strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            result_df[col] = result_df[col].fillna(fill_value)
    
    return result_df

def create_cleaning_pipeline(dataframe, operations):
    """
    Apply multiple cleaning operations in sequence
    """
    result_df = dataframe.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_outliers':
            result_df = remove_outliers_iqr(result_df, operation['column'], 
                                          operation.get('threshold', 1.5))
        elif operation['type'] == 'normalize':
            result_df = normalize_column_zscore(result_df, operation['column'])
        elif operation['type'] == 'scale':
            result_df = min_max_normalize(result_df, operation['column'],
                                        operation.get('new_min', 0),
                                        operation.get('new_max', 1))
        elif operation['type'] == 'handle_missing':
            result_df = handle_missing_values(result_df,
                                            operation.get('strategy', 'mean'),
                                            operation.get('columns'))
    
    return result_dfimport pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, convert data types,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning complete. Cleaned data saved to {output_file}")
        print(f"Original shape: {df.shape}, Duplicates removed: {len(df) - len(df.drop_duplicates())}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        print("Data cleaning summary:")
        print(f"Total rows: {len(cleaned_df)}")
        print(f"Total columns: {len(cleaned_df.columns)}")
        print("\nColumn data types:")
        print(cleaned_df.dtypes)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def get_dataset_statistics(df):
    stats_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'count': df[col].count(),
            'missing': df[col].isnull().sum()
        }
    return stats_dictimport pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output file
        missing_strategy (str): Strategy for handling missing values
            'mean' - fill with column mean (numeric only)
            'median' - fill with column median (numeric only)
            'mode' - fill with most frequent value
            'drop' - drop rows with missing values
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        print(f"Removed {original_rows - len(df)} duplicate rows")
        
        if missing_strategy == 'drop':
            df_cleaned = df.dropna()
            print(f"Removed {len(df) - len(df_cleaned)} rows with missing values")
        else:
            df_cleaned = df.copy()
            for column in df_cleaned.columns:
                if df_cleaned[column].isnull().any():
                    null_count = df_cleaned[column].isnull().sum()
                    
                    if missing_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                        fill_value = df_cleaned[column].mean()
                        df_cleaned[column].fillna(fill_value, inplace=True)
                        print(f"Filled {null_count} missing values in '{column}' with mean: {fill_value:.2f}")
                    
                    elif missing_strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                        fill_value = df_cleaned[column].median()
                        df_cleaned[column].fillna(fill_value, inplace=True)
                        print(f"Filled {null_count} missing values in '{column}' with median: {fill_value:.2f}")
                    
                    elif missing_strategy == 'mode':
                        fill_value = df_cleaned[column].mode()[0]
                        df_cleaned[column].fillna(fill_value, inplace=True)
                        print(f"Filled {null_count} missing values in '{column}' with mode: {fill_value}")
                    
                    else:
                        print(f"Column '{column}' has {null_count} missing values, but strategy '{missing_strategy}' not applicable")
        
        print(f"Final cleaned data shape: {df_cleaned.shape}")
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('Dataframe is empty or None')
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    for column in df.columns:
        null_percentage = df[column].isnull().mean() * 100
        if null_percentage > 50:
            validation_results['warnings'].append(f"Column '{column}' has {null_percentage:.1f}% missing values")
        
        unique_values = df[column].nunique()
        if unique_values == 1:
            validation_results['warnings'].append(f"Column '{column}' has only one unique value")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, np.nan, 15.2, 15.2, np.nan, 20.1],
        'category': ['A', 'B', 'A', 'A', 'B', 'C']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', missing_strategy='mean')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print(f"Validation results: {validation}")import pandas as pd
import numpy as np
from scipy import stats

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

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
        
        cleaned_file = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file, index=False)
        return f"Cleaned data saved to: {cleaned_file}"
    
    except Exception as e:
        return f"Error during cleaning: {str(e)}"

if __name__ == "__main__":
    result = clean_dataset('sample_data.csv')
    print(result)