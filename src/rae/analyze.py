"""Tabulate trajectory-regularization metrics for a saved candidate.

Reads a run_dir (that GEPA wrote via `--run-dir`), re-runs adapter.evaluate()
on a problem set, and prints the transition matrix and derived rates. No new
abstractions -- every field used here is already in the adapter's RolloutOutput
dict (`draft_score`, `final_score`, `advice_skipped`).

Usage:
    python -m rae.analyze --run-dir runs/exp3_full --split holdout \
        --arena math_zpd --best-idx 6

Transition classes (per example):
  W->R  repair
  R->R  preservation
  R->W  OVER-ADVISING regression
  W->W  failed repair
  plus NO_ADVICE precision/recall.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .actor_client import ActorClient, ActorConfig
from .arenas import math_zpd, rule_arena_taxes
from .evaluator import extract_signed_amount, score_response
from .gepa_adapter import CompoundProgramAdapter
from .scorers import score_math


_ARENAS = {
    "taxes": {
        "load": lambda: rule_arena_taxes.load_problems(0),
        "split": rule_arena_taxes.split_train_dev_holdout,
        "score": score_response,
        "extract": extract_signed_amount,
    },
    "math_zpd": {
        "load": lambda: math_zpd.load_problems(pass_lo=0.15, pass_hi=0.50),
        "split": math_zpd.split_train_dev_holdout,
        "score": score_math,
        "extract": lambda s: s,
    },
}


def _load_candidate(run_dir: Path, best_idx: int | None) -> dict:
    cands = json.loads((run_dir / "candidates.json").read_text())
    if best_idx is None:
        best_idx = len(cands) - 1
    return cands[best_idx]


def _classify(draft_correct: bool, final_correct: bool, advice_skipped: bool) -> str:
    d, f = ("R" if draft_correct else "W", "R" if final_correct else "W")
    key = f"{d}->{f}"
    if advice_skipped:
        key += "_NA"  # NO_ADVICE pathway
    return key


def _pct(num: int, den: int) -> str:
    return f"{(num / den * 100 if den else 0):5.1f}% ({num}/{den})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arena", choices=list(_ARENAS), required=True)
    p.add_argument(
        "--split",
        choices=["train", "dev", "holdout", "all"],
        default="holdout",
    )
    p.add_argument("--train-n", type=int, default=30)
    p.add_argument("--dev-n", type=int, default=15)
    p.add_argument("--holdout-n", type=int, default=15)
    p.add_argument(
        "--best-idx",
        type=int,
        default=None,
        help="Which candidate index to evaluate (default: last).",
    )
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument(
        "--full-scaffold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    cfg = _ARENAS[args.arena]
    problems = cfg["load"]()
    splits = {}
    splits["train"], splits["dev"], splits["holdout"] = cfg["split"](
        problems,
        train_n=args.train_n,
        dev_n=args.dev_n,
        holdout_n=args.holdout_n,
    )
    splits["all"] = splits["train"] + splits["dev"] + splits["holdout"]
    batch = splits[args.split]

    candidate = _load_candidate(args.run_dir, args.best_idx)
    print(f"[analyze] run_dir={args.run_dir}")
    print(f"[analyze] arena={args.arena}  split={args.split}  n={len(batch)}")
    print(f"[analyze] candidate components={list(candidate.keys())}")

    adapter = CompoundProgramAdapter(
        actor=ActorClient(ActorConfig()),
        score_fn=cfg["score"],
        extract_fn=cfg["extract"],
        full_scaffold=args.full_scaffold,
        max_workers=args.max_workers,
    )
    eb = adapter.evaluate(batch, candidate, capture_traces=False)

    # Tabulate
    classes: Counter[str] = Counter()
    advice_emitted = 0
    no_advice = 0
    preserved = 0  # R --NO_ADVICE--> R
    missed = 0  # W --NO_ADVICE--> W
    for out in eb.outputs:
        d = bool(out.get("draft_score", 0.0))
        f = bool(out.get("final_score", 0.0))
        na = bool(out.get("advice_skipped", False))
        classes[_classify(d, f, na)] += 1
        if na:
            no_advice += 1
            if d and f:
                preserved += 1
            elif (not d) and (not f):
                missed += 1
        else:
            advice_emitted += 1

    n = len(eb.outputs)
    acc_draft = sum(o.get("draft_score", 0.0) for o in eb.outputs) / max(n, 1)
    acc_final = sum(o.get("final_score", 0.0) for o in eb.outputs) / max(n, 1)

    n_W2R = sum(v for k, v in classes.items() if k.startswith("W->R"))
    n_R2W = sum(v for k, v in classes.items() if k.startswith("R->W"))
    n_R2R = sum(v for k, v in classes.items() if k.startswith("R->R"))
    n_W2W = sum(v for k, v in classes.items() if k.startswith("W->W"))
    n_R2W_adv = classes.get("R->W", 0)  # R->W without NO_ADVICE: advice was emitted

    print()
    print(
        f"[analyze] accuracy  draft={acc_draft:.4f}  final={acc_final:.4f}  delta={acc_final - acc_draft:+.4f}"
    )
    print()
    print("[analyze] transition counts (incl. _NA suffix for NO_ADVICE short-circuit):")
    for k in sorted(classes):
        print(f"   {k:10s} {classes[k]:3d}")
    print()
    print(
        f"[analyze] net regularization  P(W->R) - P(R->W) = {(n_W2R - n_R2W) / n:+.4f}"
    )
    print(f"[analyze] repair rate          P(W->R)           = {_pct(n_W2R, n)}")
    print(f"[analyze] regression rate      P(R->W)           = {_pct(n_R2W, n)}")
    print(
        f"[analyze] over-advising (emit advice -> R->W)    = {_pct(n_R2W_adv, advice_emitted)}"
    )
    print(
        f"[analyze] NO_ADVICE precision  P(R | NO_ADVICE)  = {_pct(preserved, no_advice)}"
    )
    recall_R = preserved
    n_draft_R = n_R2R + n_R2W  # all cases where draft was R
    print(
        f"[analyze] NO_ADVICE recall     P(NO_ADVICE | R)  = {_pct(recall_R, n_draft_R)}"
    )


if __name__ == "__main__":
    main()
