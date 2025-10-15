"""Shared utilities for LLM initialization and invocation.

This module centralizes common logic used across example scripts:
- reading the OpenAI API key from the environment
- locating an available ChatOpenAI wrapper from multiple langchain packages
- creating a configured ChatOpenAI instance (or returning None)
- a small helper to invoke LLMs safely regardless of .invoke/__call__ style

Extracting these helpers removes duplication and makes tests easier to
patch/match in a single place.
"""
from __future__ import annotations

import os
from typing import Optional, Any


def get_openai_api_key() -> Optional[str]:
    """Return the OpenAI API key from the environment if set.

    Centralized so tests and scripts use the same lookup.
    """
    return os.environ.get("OPENAI_API_KEY")


def import_chat_openai():
    """Attempt to import ChatOpenAI from common langchain packages.

    Returns the class if available, otherwise None.
    """
    try:
        # preferred: community package
        from langchain_community.llms import ChatOpenAI  # type: ignore

        return ChatOpenAI
    except Exception:
        pass

    try:
        import langchain_openai as _lc_openai  # type: ignore

        return getattr(_lc_openai, "ChatOpenAI", None)
    except Exception:
        pass

    try:
        from langchain.llms import ChatOpenAI  # type: ignore

        return ChatOpenAI
    except Exception:
        return None


def make_chat_llm(openai_api_key: Optional[str] = None, model_name: str = "gpt-4o", temperature: float = 0) -> Optional[Any]:
    """Create and return a ChatOpenAI instance, or None if unavailable.

    The function is conservative: it returns None on import failures or
    if the API key is missing. Tests can monkeypatch this function to
    return a dummy LLM.
    """
    if openai_api_key is None:
        openai_api_key = get_openai_api_key()

    ChatOpenAI = import_chat_openai()
    if ChatOpenAI is None or not openai_api_key:
        return None

    try:
        return ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key, model_name=model_name)
    except Exception:
        # If the real class constructor changes, return None rather than fail
        return None


def invoke_llm_safely(llm: Any, prompt: str):
    """Invoke an LLM object in a safe, tolerant way.

    Many wrappers expose different call styles: some provide `.invoke()`;
    others implement `__call__`. This helper tries `.invoke()` first and
    falls back to calling the object. Exceptions are caught and a
    descriptive string is returned so callers don't need to duplicate
    try/except logic.
    """
    try:
        if hasattr(llm, "invoke"):
            return llm.invoke(prompt)
        return llm(prompt)
    except Exception as e:
        return f"LLM invocation failed: {e}"
