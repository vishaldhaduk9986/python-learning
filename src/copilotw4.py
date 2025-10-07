
import requests

def parse_full_name(s):
    """Parses 'First Last' and returns a tuple (first, last)"""
    try:
        # split on any whitespace and collapse multiple spaces
        parts = s.strip().split(None, 1)
        if len(parts) != 2:
            raise ValueError
        first, last = parts
    except ValueError:
        raise ValueError("Input must include a space separating first and last name")
    return first, last

def calculate_ratio(a, b):
    """Returns a/b, raises ZeroDivisionError if b is zero"""
    return a / b

def get_status_from_api(url):
    """Calls a GET endpoint and returns status code"""
    resp = requests.get(url)
    return resp.status_code
