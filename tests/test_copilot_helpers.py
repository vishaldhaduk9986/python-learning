import sys
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.copilot_helpers import generate_pr_summary, generate_review_feedback, apply_inline_suggestion, ValidationStep


def test_generate_pr_summary_contains_sections():
    changes = ["Change A", "Change B"]
    validations = [ValidationStep("Run tests", "pytest -q")]
    text = generate_pr_summary(changes, author="dev", impact_level="low", validations=validations)
    assert "Summary" in text
    assert "Changes:" in text
    assert "Validation" in text


def test_generate_review_feedback_basic():
    pr_text = "This PR adds a database migration and uses gpt-4"
    comments = generate_review_feedback(pr_text)
    # We expect at least two heuristic comments (DB + heavy model)
    assert any("DB migrations" in c["comment"] for c in comments)
    assert any("hosted inference" in c["comment"] for c in comments)


def test_apply_inline_suggestion_simple_replace():
    content = "hello world"
    suggestion = {"find": "world", "replace": "universe"}
    out = apply_inline_suggestion(content, suggestion)
    assert out == "hello universe"
