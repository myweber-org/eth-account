
import pandas as pd

def clean_dataframe(df, column, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric threshold in a specified column.
    Returns a new DataFrame.
    """
    if keep_above:
        filtered_df = df[df[column] > threshold].copy()
    else:
        filtered_df = df[df[column] <= threshold].copy()

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_duplicates(df, subset=None):
    """
    Removes duplicate rows from the DataFrame.
    If subset is provided, only considers those columns for identifying duplicates.
    """
    cleaned_df = df.drop_duplicates(subset=subset, keep='first').copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def main():
    # Example usage
    data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
            'Score': [85, 92, 78, 45, 85],
            'Age': [25, 30, 35, 40, 25]}
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Clean based on score threshold
    high_scorers = clean_dataframe(df, 'Score', 80, keep_above=True)
    print("\nDataFrame with scores > 80:")
    print(high_scorers)

    # Remove duplicates
    unique_df = remove_duplicates(df, subset=['Name', 'Score'])
    print("\nDataFrame after removing duplicates (based on Name and Score):")
    print(unique_df)

if __name__ == "__main__":
    main()