import re
import string

def normalize_whitespace(text):
    """Replace multiple whitespace characters with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_punctuation(text):
    """Remove all punctuation characters from the text."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def clean_text(text, remove_punct=False):
    """Main cleaning function. Normalizes whitespace and optionally removes punctuation."""
    cleaned = normalize_whitespace(text)
    if remove_punct:
        cleaned = remove_punctuation(cleaned)
    return cleaned

def tokenize_text(text, lowercase=True):
    """Tokenize text into words. Optionally converts to lowercase."""
    if lowercase:
        text = text.lower()
    tokens = text.split()
    return tokens

def process_document(document, remove_punct=True, lowercase=True):
    """Process a document through the full cleaning and tokenization pipeline."""
    cleaned = clean_text(document, remove_punct=remove_punct)
    tokens = tokenize_text(cleaned, lowercase=lowercase)
    return tokens
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df)
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_stats = calculate_statistics(cleaned_df, column)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            cleaned_stats = calculate_statistics(cleaned_df, column)
            
            all_stats[column] = {
                'original': original_stats,
                'cleaned': cleaned_stats,
                'removed_count': original_stats['count'] - cleaned_stats['count']
            }
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    # Introduce some outliers
    sample_data['value'][95] = 500
    sample_data['value'][96] = -200
    sample_data['score'][97] = 150
    sample_data['score'][98] = -50
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics:")
    print("Value - Mean:", df['value'].mean(), "Std:", df['value'].std())
    print("Score - Mean:", df['score'].mean(), "Std:", df['score'].std())
    
    cleaned_df, stats = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print("Value - Mean:", cleaned_df['value'].mean(), "Std:", cleaned_df['value'].std())
    print("Score - Mean:", cleaned_df['score'].mean(), "Std:", cleaned_df['score'].std())
    
    print("\nRemoved outliers per column:")
    for col, col_stats in stats.items():
        print(f"{col}: {col_stats['removed_count']} outliers removed")