
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def main():
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 15, 18, 20, 22, 25, 28, 30, 100, 105, 110, 5, 8, 35, 40, 45, 50, 200]
    }
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['value'])
    print("\nCleaned data:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, ['id', 'value'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    main()