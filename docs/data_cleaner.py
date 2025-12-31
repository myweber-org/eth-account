
import re

def clean_string(text):
    """
    Clean and normalize a string by:
    - Removing leading/trailing whitespace
    - Converting to lowercase
    - Removing extra spaces between words
    - Removing non-alphanumeric characters except basic punctuation
    """
    if not isinstance(text, str):
        return text

    # Remove leading/trailing whitespace
    text = text.strip()

    # Convert to lowercase
    text = text.lower()

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-alphanumeric characters except spaces, periods, commas, and hyphens
    text = re.sub(r'[^a-z0-9\s.,-]', '', text)

    return text

def normalize_phone_number(phone):
    """
    Normalize a phone number string by removing all non-digit characters.
    Returns the cleaned digits or None if no digits are found.
    """
    if not isinstance(phone, str):
        return None

    digits = re.sub(r'\D', '', phone)

    return digits if digits else None

def validate_email(email):
    """
    Basic email validation using a regular expression.
    Returns True if the email format is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))