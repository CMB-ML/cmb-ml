from hashlib import sha256


def string_to_seed(input_string: str) -> int:
    """
    Convert a string to a seed using the SHA-256 hash.
    
    Args:
        input_string (str): The input string.

    Returns:
        int: The seed.
    """
    hash_object = sha256(input_string.encode())
    # Convert the hash to an integer
    hash_integer = int(hash_object.hexdigest(), 16)
    # Reduce the size to fit into expected seed range of ints (for numpy/pysm3)
    seed = hash_integer % (2**32)
    return seed
