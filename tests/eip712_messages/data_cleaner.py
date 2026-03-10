import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original shape: {df.shape}")
        
        df.replace(['', 'NA', 'N/A', 'null', 'NULL'], np.nan, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        df.drop_duplicates(inplace=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def z_score_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean = data[column].mean()
    std = data[column].std()
    
    if std == 0:
        return data[column]
    
    normalized = (data[column] - mean) / std
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    a, b = feature_range
    normalized = a + ((data[column] - min_val) * (b - a)) / (max_val - min_val)
    return normalized

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_percentage = data.isnull().sum() / len(data) * 100
    high_missing_cols = missing_percentage[missing_percentage > threshold].index.tolist()
    
    return {
        'missing_percentage': missing_percentage,
        'high_missing_columns': high_missing_cols,
        'total_missing': data.isnull().sum().sum()
    }

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {
        'original_rows': len(data),
        'outliers_removed': {},
        'normalized_columns': []
    }
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            try:
                filtered_data, removed = remove_outliers_iqr(cleaned_data, col, outlier_factor)
                cleaning_report['outliers_removed'][col] = removed
                cleaned_data = filtered_data
                
                if normalize_method == 'zscore':
                    cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
                elif normalize_method == 'minmax':
                    cleaned_data[f'{col}_normalized'] = min_max_normalize(cleaned_data, col)
                
                cleaning_report['normalized_columns'].append(col)
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                continue
    
    cleaning_report['final_rows'] = len(cleaned_data)
    cleaning_report['rows_removed'] = cleaning_report['original_rows'] - cleaning_report['final_rows']
    
    missing_info = detect_missing_patterns(cleaned_data)
    cleaning_report.update(missing_info)
    
    return cleaned_data, cleaning_report