"""Score extraction parity with advisor_models/rule_arena/config.compute_score.

These tests pin the regex behavior so refactors cannot drift our numbers away
from the published Advisor Models baselines.
"""

from rae.evaluator import extract_signed_amount, score_response


def test_extract_owed_simple():
    assert extract_signed_amount("The total tax owed is $1,234.") == 1234.0


def test_extract_overpaid_negative_sign():
    assert extract_signed_amount("The total tax overpaid is $500.") == -500.0


def test_extract_with_decimals():
    assert extract_signed_amount("The total tax owed is $1,234.56.") == 1234.56


def test_extract_handles_markdown_bold():
    assert extract_signed_amount("**The total tax owed is $42.**") == 42.0


def test_extract_uses_last_match_when_chain_of_thought_mentions_intermediate():
    text = (
        "Working: The total tax owed is $100. After credits we recompute. "
        "Final: The total tax owed is $250."
    )
    assert extract_signed_amount(text) == 250.0


def test_extract_returns_none_when_missing():
    assert extract_signed_amount("I don't know") is None
    assert extract_signed_amount("") is None
    assert extract_signed_amount(None) is None  # type: ignore[arg-type]


def test_score_correct():
    assert score_response("The total tax owed is $99.", 99.0) == 1.0
    assert score_response("The total tax owed is $99.", "99") == 1.0


def test_score_incorrect_returns_zero():
    assert score_response("The total tax owed is $100.", 99.0) == 0.0


def test_score_overpaid_sign_match():
    assert score_response("The total tax overpaid is $500.", -500.0) == 1.0


def test_score_no_extraction_returns_zero():
    assert score_response("I refuse to answer", 100.0) == 0.0
    assert score_response("", 0.0) == 0.0
