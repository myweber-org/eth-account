
import pandas as pd
import re

def clean_dataframe(df, text_column='text'):
    """
    Clean a DataFrame by removing duplicates and normalizing text.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase and remove extra whitespace
    def normalize_text(text):
        if pd.isna(text):
            return text
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df_clean[text_column] = df_clean[text_column].apply(normalize_text)
    
    return df_clean

def save_cleaned_data(df, output_path='cleaned_data.csv'):
    """
    Save cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data = {'text': ['Hello World  ', 'hello world', 'Python   Code', 'python code']}
    df = pd.DataFrame(data)
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    save_cleaned_data(cleaned_df)