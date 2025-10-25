"""Lightweight Copilot-style PR helpers
-------------------------------------

This module provides simple, testable utilities that emulate the kind
of PR summary and review-feedback text Copilot might generate. It's
designed for local automation in demos and tests (no network calls).

Functions:
- generate_pr_summary(changes, author, impact_level, validations)
- generate_review_feedback(pr_text)
- apply_inline_suggestion(file_content, suggestion)

These helpers are intentionally deterministic and small so they are
safe to run in CI or on low-memory hosts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import textwrap
import re


@dataclass
class ValidationStep:
    """A single validation step to be performed after the change."""
    description: str
    command: Optional[str] = None


def generate_pr_summary(changes: List[str], author: str, impact_level: str, validations: List[ValidationStep]) -> str:
    """Return a professional PR summary describing the change.

    - changes: list of short lines describing the edits
    - author: human-readable author string
    - impact_level: one of 'low', 'medium', 'high'
    - validations: a list of ValidationStep describing how to verify

    The returned string is suitable for pasting into a PR description.
    """
    bulleted = "\n".join(f"- {c}" for c in changes)
    validation_text = "\n".join(
        f"- {v.description}" + (f" (run: `{v.command}`)" if v.command else "") for v in validations
    )

    body = textwrap.dedent(f"""
    Summary
    -------
    Author: {author}
    Impact: {impact_level}

    Changes:
    {bulleted}

    Risks and mitigations
    ---------------------
    - {('Low risk: internal-only change' if impact_level=='low' else 'Requires smoke testing and review')}

    Validation
    ----------
    {validation_text}

    Notes
    -----
    This PR was prepared using a local Copilot-helper module that emits
    concise guidance for reviewers. Please run the validations and add
    any environment-specific checks as needed.
    ""
    )
    return body


def generate_review_feedback(pr_text: str) -> List[Dict[str, str]]:
    """Generate a short list of review comments based on the PR text.

    This is a heuristic function useful for prototyping automated
    review comments. Each comment is a dict with 'path' and 'comment'.
    """
    comments: List[Dict[str, str]] = []
    # If PR mentions 'database' warn about migrations
    if re.search(r"\bdatabase\b|migration|migrate", pr_text, re.I):
        comments.append({"path": "README.md", "comment": "DB migrations need clear rollback steps and migration tests."})

    # If heavy model names are present, recommend hosted fallback
    if re.search(r"gpt-4|gpt-4o|llama|bloom", pr_text, re.I):
        comments.append({"path": "src/utils.py", "comment": "Consider adding a hosted inference fallback and avoid loading large models by default to support low-memory hosts."})

    # Generic suggestions
    comments.append({"path": "", "comment": "Add a short list of integration tests that exercise the happy path and one failure path."})
    return comments


def apply_inline_suggestion(file_content: str, suggestion: Dict[str, str]) -> str:
    """Apply a simple inline suggestion to a file content string.

    suggestion is a dict with:
    - 'find': substring or regex to find
    - 'replace': replacement string

    This function will attempt a single-pass substitution and return
    the modified content. It never writes files — callers should write
    the result if desired.
    """
    find = suggestion.get("find")
    replace = suggestion.get("replace", "")
    # If 'find' looks like a regex (contains \b or ^ or $), use sub
    try:
        if any(ch in find for ch in ["\\b", "^", "$", "(?", "\\s"]):
            return re.sub(find, replace, file_content)
        return file_content.replace(find, replace)
    except Exception:
        # On any error, return original content unchanged
        return file_content


if __name__ == "__main__":
    # Demo mode: print a sample PR summary and a few review comments
    sample_changes = [
        "Add `src/copilot_helpers.py` for generating PR summaries and review feedback",
        "Add unit tests for the helper functions",
        "Document usage in README",
    ]
    sample_validations = [
        ValidationStep("Run unit tests", "pytest -q"),
        ValidationStep("Open htmlcov/index.html and inspect changed files"),
    ]
    print(generate_pr_summary(sample_changes, author="automation-bot", impact_level="low", validations=sample_validations))
    print("\n--- review feedback (simulated) ---\n")
    for c in generate_review_feedback(' '.join(sample_changes)):
        print(f"{c['path']}: {c['comment']}")
