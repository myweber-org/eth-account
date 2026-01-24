import re
import string

def remove_punctuation(text):
    """Remove all punctuation from the input string."""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_whitespace(text):
    """Replace multiple whitespace characters with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text, remove_punct=True, normalize_ws=True):
    """Clean text by optionally removing punctuation and normalizing whitespace."""
    if remove_punct:
        text = remove_punctuation(text)
    if normalize_ws:
        text = normalize_whitespace(text)
    return text.lower()

def tokenize_text(text, delimiter=' '):
    """Split text into tokens based on the specified delimiter."""
    return text.split(delimiter)

def process_text_list(text_list, **kwargs):
    """Apply cleaning functions to a list of text strings."""
    return [clean_text(text, **kwargs) for text in text_list]