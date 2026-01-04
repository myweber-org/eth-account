import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Standardize column names: strip whitespace and convert to lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Convert date columns if they exist (common patterns)
    date_columns = [col for col in df.columns if 'date' in col]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass  # Keep as-is if conversion fails

    # Fill numeric missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)

    # Remove rows where all values are NaN
    df.dropna(how='all', inplace=True)

    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {original_shape[0] - cleaned_shape[0]} duplicate/empty rows.")

    if output_path:
        try:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    return df

def summarize_data(df):
    """Generate a basic summary of the DataFrame."""
    if df is None or df.empty:
        print("No data to summarize.")
        return

    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'date_columns': list(df.select_dtypes(include=['datetime64']).columns),
        'missing_values': df.isnull().sum().sum()
    }

    print("\n=== Data Summary ===")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    if summary['numeric_columns']:
        print("\nNumeric Columns Statistics:")
        print(df[summary['numeric_columns']].describe().round(2))

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = clean_csv_data(input_file, output_file)
    if cleaned_df is not None:
        summarize_data(cleaned_df)