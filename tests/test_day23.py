import sys
from pathlib import Path
# ensure repo root is on sys.path for imports like `src.xxx`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import src.day23 as day23


def test_verify_api_key_valid():
    # valid key should be returned
    returned = day23.verify_api_key(api_key=day23.VALID_API_KEY)
    assert returned == day23.VALID_API_KEY


def test_verify_api_key_invalid():
    with pytest.raises(Exception) as excinfo:
        day23.verify_api_key(api_key="bad-key")
    # expect HTTPException-like with status attribute or message
    assert "Invalid or missing API Key" in str(excinfo.value)


def test_qa_endpoint_direct_call():
    # call the path operation directly (avoids rate-limiter/startup complexities)
    import asyncio

    result = asyncio.run(day23.qa_endpoint(None, "hello", day23.VALID_API_KEY))
    assert isinstance(result, dict)
    assert "Answer generated for: hello" in result.get("response")
