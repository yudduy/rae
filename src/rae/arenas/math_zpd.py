"""Phase-0 ZPD MATH arena.

Loads from the existing decomposition-mve phase0 K=64 actor evaluation
(`decomposition-mve/results/runs/decomp_mve/phase0/actor_eval_k64.jsonl`)
and filters to "in-zone" problems where Qwen2.5-7B sometimes succeeds
(default: 0.15 <= pass_rate <= 0.50). These are the problems where a scaffold
intervention has a chance to push performance up -- unlike fail@64 (floor) or
near-100% (ceiling).

Gold answer is extracted from the `\\boxed{...}` in the `solution` field
because the dataset's `answer` field is empty for these records.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_PHASE0_JSONL = Path(
    "/Users/duy/Documents/build/dc/decomposition-mve/results/runs/decomp_mve/phase0/actor_eval_k64.jsonl"
)

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


@dataclass(frozen=True)
class MathProblem:
    instance_id: str
    level: str
    subject: str
    pass_rate_baseline: float  # actor's K=64 pass rate (signal-richness indicator)
    question: str
    ground_truth: str  # raw boxed expression (kept name-compatible with TaxProblem)


def _extract_boxed(solution: str) -> str | None:
    """Extract the LAST \\boxed{...} content from a MATH solution."""
    if not solution:
        return None
    matches = list(_BOXED_RE.finditer(solution))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def load_problems(
    *,
    pass_lo: float = 0.15,
    pass_hi: float = 0.50,
    max_n: int | None = None,
    src: Path = DEFAULT_PHASE0_JSONL,
) -> list[MathProblem]:
    if not src.exists():
        raise FileNotFoundError(f"Phase-0 actor eval file missing: {src}")

    out: list[MathProblem] = []
    with src.open() as f:
        for line in f:
            d = json.loads(line)
            pr = float(d.get("pass_rate", 0.0))
            if not (pass_lo <= pr <= pass_hi):
                continue
            gold = _extract_boxed(d.get("solution", ""))
            if not gold:
                continue
            out.append(
                MathProblem(
                    instance_id=str(d["instance_id"]),
                    level=str(d.get("level", "")),
                    subject=str(d.get("subject", "")),
                    pass_rate_baseline=pr,
                    question=str(d["problem"]),
                    ground_truth=gold,
                )
            )
            if max_n is not None and len(out) >= max_n:
                break
    return out


def split_train_dev_holdout(
    problems: Iterable[MathProblem],
    *,
    train_n: int,
    dev_n: int,
    holdout_n: int,
    seed: int = 42,
) -> tuple[list[MathProblem], list[MathProblem], list[MathProblem]]:
    import random

    items = list(problems)
    rng = random.Random(seed)
    rng.shuffle(items)
    return (
        items[:train_n],
        items[train_n : train_n + dev_n],
        items[train_n + dev_n : train_n + dev_n + holdout_n],
    )
