from dotenv import load_dotenv
load_dotenv()
import os

def get_api_key(api_key_name: str) -> str:
    """Retrieves an API key from environment variables."""
    api_key = os.environ.get(api_key_name)
    if not api_key:
        raise ValueError(f"API key '{api_key_name}' not found in environment variables.")
    return api_key