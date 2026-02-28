
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()

    if remove_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate rows.")

    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()

    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))

            if case_normalization == 'lower':
                df_clean[col] = df_clean[col].str.lower()
            elif case_normalization == 'upper':
                df_clean[col] = df_clean[col].str.upper()
            elif case_normalization == 'title':
                df_clean[col] = df_clean[col].str.title()

            print(f"Cleaned column: {col}")

    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    valid_count = df['email_valid'].sum()
    total_count = df.shape[0]

    print(f"Valid emails: {valid_count}/{total_count}")
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'alice', 'Bob  ', 'Charlie', 'Alice'],
        'email': ['alice@example.com', 'invalid-email', 'bob@test.org', 'charlie@demo.net', 'alice@example.com'],
        'age': [25, 25, 30, 35, 25]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataframe(df, columns_to_clean=['name', 'email'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)

    validated_df = validate_email_column(cleaned_df, 'email')
    print("\nDataFrame with email validation:")
    print(validated_df[['name', 'email', 'email_valid']])