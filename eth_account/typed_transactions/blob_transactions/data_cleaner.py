
import re

def clean_string(input_string):
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    cleaned = input_string.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.lower()
    
    return cleaned

def remove_special_characters(input_string, keep_chars=""):
    pattern = f"[^a-zA-Z0-9{re.escape(keep_chars)}\s]"
    return re.sub(pattern, '', input_string)