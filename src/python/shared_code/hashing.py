import hashlib


def hash_string(input_string: str) -> str:
    """
    Hash a string using SHA-256 and return the hexadecimal representation.
    
    Args:
        input_string (str): The string to hash.
        
    Returns:
        str: The SHA-256 hash of the input string in hexadecimal format.
    """
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    return sha256_hash