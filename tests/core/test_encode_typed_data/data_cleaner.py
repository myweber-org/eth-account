
def filter_valid_entries(data_list, required_keys):
    """
    Returns a new list containing only dictionaries that have all specified keys
    and where none of the required key values are None or empty strings.
    """
    if not isinstance(data_list, list):
        raise TypeError("Input must be a list")
    if not isinstance(required_keys, list):
        raise TypeError("Required keys must be a list")

    filtered_data = []
    for entry in data_list:
        if not isinstance(entry, dict):
            continue

        is_valid = True
        for key in required_keys:
            if key not in entry or entry[key] in (None, ""):
                is_valid = False
                break

        if is_valid:
            filtered_data.append(entry)

    return filtered_datadef remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Remove rows where critical columns are still null
        critical_columns = ['id', 'timestamp', 'value']
        existing_critical = [col for col in critical_columns if col in df.columns]
        if existing_critical:
            initial_count = len(df)
            df = df.dropna(subset=existing_critical)
            print(f"Removed {initial_count - len(df)} rows with null critical columns")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    cleaned_df = clean_csv_data(input_csv, output_csv)