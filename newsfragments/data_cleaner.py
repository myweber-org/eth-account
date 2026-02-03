import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[col].index[z_scores > 3]
        df.loc[outliers, col] = np.nan
        print(f"Removed {len(outliers)} outliers from {col}")
    
    df_cleaned = df.dropna()
    print(f"Cleaned shape: {df_cleaned.shape}")
    
    for col in numeric_cols:
        if col in df_cleaned.columns:
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            if col_max > col_min:
                df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("sample_data.csv")
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning complete. Saved to cleaned_data.csv")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and standardizing formats.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Display initial info
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        # Fill numeric missing values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"\nCleaning completed:")
        print(f"- Removed {duplicates_removed} duplicate rows")
        print(f"- Final shape: {df.shape}")
        print(f"- Saved to: {output_path}")
        print(f"Missing values after cleaning:")
        print(df.isnull().sum())
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None, None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Warning: Column '{col}' contains infinite values")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, None, 35, 40, 28],
        'Salary': [50000, 60000, None, 75000, 48000],
        'Department': ['HR', 'IT', 'IT', None, 'Finance']
    }
    
    # Create sample CSV
    pd.DataFrame(sample_data).to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df, output_file = clean_csv_data('sample_data.csv')
    
    if cleaned_df is not None:
        print("\nFirst few rows of cleaned data:")
        print(cleaned_df.head())
        
        # Validate the cleaned data
        is_valid = validate_dataframe(cleaned_df, ['name', 'age', 'salary'])
        print(f"\nData validation: {'Passed' if is_valid else 'Failed'}")