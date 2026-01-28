import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            if missing_strategy == 'mean':
                # Fill numeric columns with mean
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                # Fill non-numeric with mode
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            elif missing_strategy == 'drop':
                df = df.dropna()
            else:
                raise ValueError("Strategy must be 'mean' or 'drop'")
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows.")
            df = df.drop_duplicates()
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}. New shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    clean_csv_data('raw_data.csv', 'cleaned_data.csv', missing_strategy='mean')