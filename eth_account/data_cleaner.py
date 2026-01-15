import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
        
        print(f"After outlier removal: {df.shape}")
        
        for col in numeric_cols:
            df = normalize_column(df, col)
        
        cleaned_file = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file, index=False)
        print(f"Cleaned data saved to: {cleaned_file}")
        return df
        
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    sample_data.loc[10, 'feature1'] = 500
    sample_data.loc[20, 'feature2'] = 1000
    
    sample_data.to_csv('sample_dataset.csv', index=False)
    cleaned_df = clean_dataset('sample_dataset.csv')