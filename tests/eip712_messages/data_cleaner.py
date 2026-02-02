import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Args:
        filepath: Path to the CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns_to_drop: List of column names to drop (optional)
    
    Returns:
        Cleaned DataFrame and report dictionary
    """
    try:
        df = pd.read_csv(filepath)
        original_shape = df.shape
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        missing_report = {
            'total_missing': df.isnull().sum().sum(),
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_per_column': df.isnull().sum().to_dict()
        }
        
        if missing_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif missing_strategy == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        elif missing_strategy == 'drop':
            df = df.dropna()
        
        cleaned_shape = df.shape
        rows_removed = original_shape[0] - cleaned_shape[0]
        cols_removed = original_shape[1] - cleaned_shape[1]
        
        report = {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_shape,
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_handled': missing_report,
            'dtypes': df.dtypes.to_dict(),
            'strategy_used': missing_strategy
        }
        
        return df, report
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None, unique_constraints=None):
    """
    Validate DataFrame structure and constraints.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        unique_constraints: List of columns that should have unique values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty or None')
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    if unique_constraints:
        for column in unique_constraints:
            if column in df.columns:
                duplicates = df[column].duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(
                        f'Column {column} has {duplicates} duplicate values'
                    )
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        validation_results['warnings'].append('No numeric columns found in DataFrame')
    
    return validation_results

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the file
        format: Output format ('csv', 'parquet', 'json')
    
    Returns:
        Boolean indicating success
    """
    try:
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            print(f"Unsupported format: {format}")
            return False
        
        print(f"Data successfully saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False