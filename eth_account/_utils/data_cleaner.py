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