
def clean_data(entries, key_check):
    """
    Remove entries where the specified key is missing or has a falsy value.
    
    Args:
        entries (list of dict): List of dictionaries to clean.
        key_check (str): Key to validate in each dictionary.
    
    Returns:
        list of dict: Filtered list containing only valid entries.
    """
    if not isinstance(entries, list):
        raise TypeError("Entries must be a list")
    
    cleaned = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get(key_check):
            cleaned.append(entry)
    
    return cleaned