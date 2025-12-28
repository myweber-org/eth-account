
import re

def clean_string(text):
    """
    Cleans a string by removing extra whitespace and converting to lowercase.
    
    Args:
        text (str): The input string to clean.
    
    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    # Remove leading and trailing whitespace
    cleaned = text.strip()
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Convert to lowercase
    cleaned = cleaned.lower()
    
    return cleaned