
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): The list from which duplicates should be removed.
    
    Returns:
        list: A new list with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, default=0):
    """
    Clean a list of numeric values by converting non-numeric entries to default.
    
    Args:
        values (list): List of values to clean.
        default (int/float): Default value for non-numeric entries.
    
    Returns:
        list: Cleaned list of numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    
    Args:
        data (list): Data to filter.
        threshold: Minimum value to include.
        key (callable): Function to extract value from each element.
    
    Returns:
        list: Filtered data.
    """
    if key is None:
        key = lambda x: x
    
    return [item for item in data if key(item) >= threshold]

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    mixed_data = [1, "2", 3.5, "invalid", 4]
    print("Numeric cleaned:", clean_numeric_data(mixed_data))
    
    values = [10, 5, 20, 3, 15]
    print("Filtered (>=10):", filter_by_threshold(values, 10))