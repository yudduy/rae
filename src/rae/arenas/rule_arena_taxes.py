"""RuleArena Taxes loader.

Reads `references/advisor-models/advisor_models/rule_arena/RuleArena/tax/synthesized_problems/comp_{0,1,2}.json`,
builds the prompt via build_prompt() copied from advisor_models/rule_arena/config.py,
and computes the ground-truth tax via RuleArena/tax/micro_evaluation.compute_answer.

We import the RuleArena tax modules by path-injecting the vendored reference
directory rather than copying their source -- they have transitive imports
(prompt.py, structured_forms.py, gen_payer.py) that are stable and verified.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
ADVISOR_REF_ROOT = REPO_ROOT / "references" / "advisor-models"
RULEARENA_TAX_DIR = (
    ADVISOR_REF_ROOT / "advisor_models" / "rule_arena" / "RuleArena" / "tax"
)

# Path-inject so RuleArena/tax modules resolve their sibling imports.
if str(RULEARENA_TAX_DIR) not in sys.path:
    sys.path.insert(0, str(RULEARENA_TAX_DIR))
ADVISOR_RULE_ARENA_PKG = ADVISOR_REF_ROOT / "advisor_models" / "rule_arena"
if str(ADVISOR_RULE_ARENA_PKG) not in sys.path:
    sys.path.insert(0, str(ADVISOR_RULE_ARENA_PKG))


@dataclass(frozen=True)
class TaxProblem:
    instance_id: str
    complexity: int
    question: str
    ground_truth: float
    info_dict: dict


def _import_tax_modules():
    # Imported lazily so unit tests that mock the arena don't need RuleArena deps.
    from micro_evaluation import compute_answer  # type: ignore
    from structured_forms import TaxPayer  # type: ignore
    from config import build_prompt  # type: ignore  # advisor-models config

    return build_prompt, compute_answer, TaxPayer


def load_problems(complexity: int, *, max_n: int | None = None) -> list[TaxProblem]:
    """Load and ground-truth-compute tax problems at a given complexity level."""
    build_prompt, compute_answer, TaxPayer = _import_tax_modules()
    src = RULEARENA_TAX_DIR / "synthesized_problems" / f"comp_{complexity}.json"
    if not src.exists():
        raise FileNotFoundError(f"Synthesized problems missing: {src}")

    raw = json.loads(src.read_text())
    if max_n is not None:
        raw = raw[:max_n]

    out: list[TaxProblem] = []
    for i, entry in enumerate(raw):
        info_dict = {
            "pydantic": entry.get("pydantic", {}),
            "dict": entry.get("dict", {}),
        }
        try:
            taxpayer = TaxPayer(**info_dict["pydantic"])
            gt, _ = compute_answer(taxpayer)
            question = build_prompt(info_dict)
        except Exception as e:  # noqa: BLE001
            # Skip malformed entries; surface count to caller via len(out) < len(raw).
            print(f"[rule_arena_taxes] skipped comp_{complexity}#{i}: {e}")
            continue
        out.append(
            TaxProblem(
                instance_id=f"comp{complexity}_{i:03d}",
                complexity=complexity,
                question=question,
                ground_truth=float(gt),
                info_dict=info_dict,
            )
        )
    return out


def split_train_dev_holdout(
    problems: Iterable[TaxProblem],
    *,
    train_n: int,
    dev_n: int,
    holdout_n: int,
    seed: int = 42,
) -> tuple[list[TaxProblem], list[TaxProblem], list[TaxProblem]]:
    import random

    items = list(problems)
    rng = random.Random(seed)
    rng.shuffle(items)
    train = items[:train_n]
    dev = items[train_n : train_n + dev_n]
    holdout = items[train_n + dev_n : train_n + dev_n + holdout_n]
    return train, dev, holdout
