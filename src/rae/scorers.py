"""Score adapters per-arena.

Both arenas share the (response, ground_truth) -> float (0|1) signature, so
the GEPAAdapter can be parametrised by a single `score_fn` callable.

Tax score: regex extract + np.isclose (advisor-models parity).
Math score: math_verify (SymPy-based) over the LAST \\boxed{...} in the response.
"""

from __future__ import annotations

import re
from typing import Callable

from .evaluator import score_response as _score_taxes  # noqa: F401  (re-export)


_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _extract_last_boxed(s: str) -> str | None:
    if not s:
        return None
    matches = list(_BOXED_RE.finditer(s))
    return matches[-1].group(1).strip() if matches else None


def score_math(response: str, gold: str) -> float:
    """math_verify-backed scorer; falls back to normalized string equality."""
    pred = _extract_last_boxed(response)
    if pred is None:
        return 0.0

    try:
        from math_verify import parse, verify  # type: ignore

        try:
            ok = bool(verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{pred}}}")))
            return 1.0 if ok else 0.0
        except Exception:
            pass
    except ImportError:
        pass

    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def _norm(s: str) -> str:
    return "".join(s.split()).replace(",", "").lower()


ScoreFn = Callable[[str, object], float]
