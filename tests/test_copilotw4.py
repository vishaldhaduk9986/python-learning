import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure repository root is on sys.path so `src` package can be imported when
# pytest changes the working directory during collection.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import copilotw4


def test_parse_full_name_normal():
    first, last = copilotw4.parse_full_name("John Doe")
    assert first == "John"
    assert last == "Doe"


def test_parse_full_name_extra_spaces():
    first, last = copilotw4.parse_full_name("  Jane   Smith  ")
    assert first == "Jane"
    assert last == "Smith"


def test_parse_full_name_single_name_raises():
    with pytest.raises(ValueError):
        copilotw4.parse_full_name("Cher")


def test_calculate_ratio_normal():
    assert copilotw4.calculate_ratio(10, 2) == 5


def test_calculate_ratio_zero_division():
    with pytest.raises(ZeroDivisionError):
        copilotw4.calculate_ratio(1, 0)


@patch("src.copilotw4.requests.get")
def test_get_status_from_api_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    status = copilotw4.get_status_from_api("http://example.test")
    assert status == 200
    mock_get.assert_called_once_with("http://example.test")


@patch("src.copilotw4.requests.get")
def test_get_status_from_api_raises(mock_get):
    mock_get.side_effect = Exception("network error")
    with pytest.raises(Exception):
        copilotw4.get_status_from_api("http://example.test")
