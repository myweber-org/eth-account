import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.
    
    Args:
        text: The input string to clean.
        keep_spaces: If True, spaces are preserved. If False, spaces are removed.
    
    Returns:
        The cleaned string containing only alphanumeric characters and optionally spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9 ]+'
    else:
        pattern = r'[^A-Za-z0-9]+'
    
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple consecutive whitespace characters with a single space.
    
    Args:
        text: The input string to normalize.
    
    Returns:
        String with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_pipeline(text: str, 
                       remove_special: bool = True,
                       normalize_ws: bool = True,
                       to_lowercase: bool = False) -> str:
    """
    Apply a series of cleaning operations to the input text.
    
    Args:
        text: The input string to process.
        remove_special: Whether to remove special characters.
        normalize_ws: Whether to normalize whitespace.
        to_lowercase: Whether to convert text to lowercase.
    
    Returns:
        The processed text after applying all specified operations.
    """
    processed = text
    
    if remove_special:
        processed = remove_special_characters(processed)
    
    if normalize_ws:
        processed = normalize_whitespace(processed)
    
    if to_lowercase:
        processed = processed.lower()
    
    return processed

def batch_clean_texts(texts: List[str], **kwargs) -> List[str]:
    """
    Apply cleaning pipeline to a list of text strings.
    
    Args:
        texts: List of strings to clean.
        **kwargs: Arguments to pass to clean_text_pipeline.
    
    Returns:
        List of cleaned text strings.
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]

def validate_email(email: str) -> bool:
    """
    Validate if a string is a properly formatted email address.
    
    Args:
        email: The string to validate as an email.
    
    Returns:
        True if the string matches email format, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def extract_hashtags(text: str) -> List[str]:
    """
    Extract all hashtags from a text string.
    
    Args:
        text: The input string containing hashtags.
    
    Returns:
        List of hashtag strings without the '#' symbol.
    """
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags

if __name__ == "__main__":
    # Example usage
    sample_text = "Hello, World!  This   is   a   test...  #python #cleaning"
    
    cleaned = clean_text_pipeline(
        sample_text, 
        remove_special=True, 
        normalize_ws=True, 
        to_lowercase=True
    )
    
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Hashtags: {extract_hashtags(sample_text)}")
    
    test_email = "user@example.com"
    print(f"Email '{test_email}' valid: {validate_email(test_email)}")