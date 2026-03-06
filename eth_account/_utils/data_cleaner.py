
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(f"Removed {len(sample_data) - len(result)} outliers")import pandas as pd
import numpy as np
import argparse
import os

def load_data(filepath):
    """Load CSV data from the given filepath."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Perform basic cleaning operations on the dataframe."""
    if df is None:
        return None
    
    original_shape = df.shape
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # Remove columns with too many missing values (threshold: 50%)
    missing_threshold = 0.5
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > missing_threshold]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
    
    print(f"Data cleaned. Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    return df

def save_clean_data(df, original_path):
    """Save cleaned data to a new file."""
    if df is None:
        return False
    
    base_name = os.path.splitext(original_path)[0]
    clean_path = f"{base_name}_cleaned.csv"
    
    try:
        df.to_csv(clean_path, index=False)
        print(f"Cleaned data saved to: {clean_path}")
        return True
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return False

def generate_summary(df):
    """Generate a summary of the cleaned dataframe."""
    if df is None:
        return
    
    print("\n=== DATA SUMMARY ===")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn data types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if missing_cols.empty:
        print("No missing values found.")
    else:
        for col, count in missing_cols.items():
            print(f"{col}: {count} missing ({count/len(df)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Clean CSV data files.')
    parser.add_argument('filepath', help='Path to the CSV file to clean')
    parser.add_argument('--summary', action='store_true', help='Generate data summary')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.filepath)
    
    if df is not None:
        # Clean data
        cleaned_df = clean_data(df)
        
        # Save cleaned data
        if cleaned_df is not None:
            save_clean_data(cleaned_df, args.filepath)
            
            # Generate summary if requested
            if args.summary:
                generate_summary(cleaned_df)

if __name__ == "__main__":
    main()