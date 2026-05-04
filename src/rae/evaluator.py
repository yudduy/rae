"""Tax-answer extraction and scoring.

Parity with advisor_models/rule_arena/config.py compute_score: regex extracts
'The total tax (owed|overpaid) is $xxx.' from the final response, converts to a
signed float (overpaid -> negative), and compares to the gold ground truth via
np.isclose. This MUST match advisor-models numerically so our results are
comparable to the published Advisor Models / static-GEPA baselines.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

_ANSWER_PATTERN = re.compile(
    r"The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?)\.?"
)


def extract_signed_amount(response_str: str) -> Optional[float]:
    """Return signed tax amount (overpaid as negative). None if not found.

    Strips markdown bold so '**The total tax owed is $5.**' is parsed.
    Uses the LAST regex match (final-line wins) to be robust to chain-of-thought
    that mentions intermediate amounts using the same phrasing.
    """
    if not response_str:
        return None
    cleaned = response_str.replace("**", "")
    matches = list(_ANSWER_PATTERN.finditer(cleaned))
    if not matches:
        return None
    m = matches[-1]
    status = m.group(1)
    value = float(m.group(2).replace(",", ""))
    return -value if status == "overpaid" else value


def score_response(response_str: str, ground_truth: float | int | str) -> float:
    """Return 1.0 if extracted == ground_truth (np.isclose), else 0.0."""
    extracted = extract_signed_amount(response_str)
    if extracted is None:
        return 0.0
    try:
        gt = float(str(ground_truth).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return 0.0
    return 1.0 if np.isclose(extracted, gt) else 0.0
