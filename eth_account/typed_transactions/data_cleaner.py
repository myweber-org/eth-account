
import pandas as pd
import numpy as np

def clean_data(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if df.isnull().sum().any():
        missing_counts = df.isnull().sum()
        print(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif fill_missing == 'drop':
            df = df.dropna()
            print("Rows with missing values have been dropped.")
        else:
            print(f"Unknown fill method: {fill_missing}. No filling performed.")
    
    print(f"Data cleaned. Original shape: {original_shape}, Cleaned shape: {df.shape}")
    return df

def validate_data(df, check_duplicates=True, check_missing=True):
    """
    Validate the cleanliness of a DataFrame.
    """
    issues = []
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows.")
    
    if check_missing:
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values.")
    
    if issues:
        print("Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Data validation passed. No duplicates or missing values found.")
        return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_data(df.copy(), fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation_result = validate_data(cleaned_df)