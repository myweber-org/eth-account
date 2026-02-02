
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list containing elements that may have duplicates.
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to integers where possible.
    
    Args:
        data_list: A list containing mixed types.
    
    Returns:
        A new list with numeric strings converted to integers.
    """
    result = []
    for item in data_list:
        if isinstance(item, str) and item.isdigit():
            result.append(int(item))
        else:
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", 5, "hello", 3]
    cleaned = remove_duplicates(sample_data)
    final_data = clean_numeric_strings(cleaned)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {final_data}")