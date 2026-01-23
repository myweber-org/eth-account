import re
import string

def clean_text(text, remove_punctuation=True, lowercase=True, remove_numbers=False):
    """
    Clean and normalize a given text string.

    Args:
        text (str): The input text to clean.
        remove_punctuation (bool): If True, remove all punctuation.
        lowercase (bool): If True, convert text to lowercase.
        remove_numbers (bool): If True, remove all digits.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    if remove_numbers:
        cleaned = re.sub(r'\d+', '', cleaned)

    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        cleaned = cleaned.translate(translator)

    if lowercase:
        cleaned = cleaned.lower()

    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned

def tokenize_text(text, delimiter=' '):
    """
    Split text into tokens based on a delimiter.

    Args:
        text (str): The input text.
        delimiter (str): The delimiter to split on.

    Returns:
        list: A list of tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split(delimiter)

if __name__ == "__main__":
    sample_text = "Hello, World! This is a TEST. 12345"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {clean_text(sample_text)}")
    print(f"Tokens: {tokenize_text(sample_text)}")