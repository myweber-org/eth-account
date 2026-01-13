
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df.reset_index(drop=True)

def main():
    data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 15, 120, 14, 16, 18, 19, 20,
                  22, 24, 26, 28, 30, 32, 34, 200, 36, 38]
    }
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['value'])
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")

if __name__ == "__main__":
    main()