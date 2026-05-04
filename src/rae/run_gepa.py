"""Run GEPA-Advisor evolution on RuleArena Taxes.

Usage:
  python -m rae.run_gepa --variant full      # 4-module compound program (north-star)
  python -m rae.run_gepa --variant actor     # actor-only baseline (Experiment 1)

Model wiring:
  Actor / Advisor / Reviser  -> Qwen2.5-7B-Instruct served by vLLM at
                                 RAE_ACTOR_BASE_URL (default localhost:8001/v1).
  Reflection LM (proposes mutations to candidate prompts) -> set via
                                 --reflection-lm. Defaults to
                                 'together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo'
                                 if TOGETHER_API_KEY is in env, else falls back
                                 to the same local Qwen2.5-7B (weak but free).

The reflection LM choice is the single biggest controllable factor for GEPA
quality (per Section 5 of arXiv:2507.19457). A strong reflection LM mutating
prompts that condition a weak Actor is the published GEPA recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gepa

from .actor_client import ActorClient, ActorConfig
from .arenas import math_zpd, rule_arena_taxes
from .evaluator import extract_signed_amount, score_response
from .gepa_adapter import CompoundProgramAdapter
from .scorers import score_math
from .seed_prompts import actor_only_seed_candidate, default_seed_candidate
from .seed_prompts_math import (
    actor_only_seed_candidate_math,
    default_seed_candidate_math,
)


# Arena registry: (loader, splitter, score_fn, extract_fn, full_seed, actor_only_seed)
_ARENAS = {
    "taxes": {
        "load": lambda: rule_arena_taxes.load_problems(0),
        "split": rule_arena_taxes.split_train_dev_holdout,
        "score": score_response,
        "extract": extract_signed_amount,
        "full_seed": default_seed_candidate,
        "actor_seed": actor_only_seed_candidate,
    },
    "math_zpd": {
        "load": lambda: math_zpd.load_problems(pass_lo=0.15, pass_hi=0.50),
        "split": math_zpd.split_train_dev_holdout,
        "score": score_math,
        "extract": lambda s: s,  # raw response excerpt is fine for feedback
        "full_seed": default_seed_candidate_math,
        "actor_seed": actor_only_seed_candidate_math,
    },
}


def _default_reflection_lm() -> str:
    if os.environ.get("TOGETHER_API_KEY"):
        return "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    return "local-vllm"  # sentinel: build a callable backed by ActorClient


def _build_reflection_lm(spec: str, *, actor: ActorClient):
    """Resolve --reflection-lm to either a litellm string or a Python callable.

    The 'local-vllm' sentinel returns a callable that re-uses our ActorClient
    (same Qwen2.5-7B at port 8001). This is weak relative to GPT-4 / Qwen2.5-72B
    but lets the loop run end-to-end without external API keys -- still useful
    as a sanity baseline.
    """
    if spec != "local-vllm":
        return spec  # gepa.optimize will wrap a string via gepa.lm.LM(litellm)

    def _local_reflect(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        # Reflection prompts can be long; give plenty of output room.
        return actor.chat(messages, temperature=0.7, max_tokens=2048)

    return _local_reflect


def _load_split(arena: str, train_n: int, dev_n: int, holdout_n: int):
    cfg = _ARENAS[arena]
    all_problems = cfg["load"]()
    if len(all_problems) < train_n + dev_n + holdout_n:
        raise SystemExit(
            f"Not enough problems for arena={arena}: "
            f"have {len(all_problems)}, need {train_n + dev_n + holdout_n}"
        )
    return cfg["split"](all_problems, train_n=train_n, dev_n=dev_n, holdout_n=holdout_n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arena", choices=list(_ARENAS), default="taxes")
    p.add_argument("--variant", choices=["actor", "full"], default="full")
    p.add_argument("--train-n", type=int, default=20)
    p.add_argument("--dev-n", type=int, default=15)
    p.add_argument("--holdout-n", type=int, default=15)
    p.add_argument("--budget", type=int, default=120, help="max_metric_calls")
    p.add_argument("--reflection-lm", default=_default_reflection_lm())
    p.add_argument("--minibatch", type=int, default=3)
    p.add_argument(
        "--module-selector", default="round_robin", choices=["round_robin", "all"]
    )
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--acceptance",
        default="strict_improvement",
        choices=["strict_improvement", "improvement_or_equal"],
        help="When the actor is at floor, 'improvement_or_equal' lets GEPA explore.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--run-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "runs"
            / time.strftime("rae_%Y%m%d_%H%M%S")
        ),
    )
    args = p.parse_args()

    cfg = _ARENAS[args.arena]
    train, dev, holdout = _load_split(
        args.arena, args.train_n, args.dev_n, args.holdout_n
    )
    print(
        f"[rae] arena={args.arena} split: train={len(train)} dev={len(dev)} holdout={len(holdout)}"
    )

    actor = ActorClient(ActorConfig())
    full_scaffold = args.variant == "full"
    seed_candidate = cfg["full_seed"]() if full_scaffold else cfg["actor_seed"]()
    print(f"[rae] variant={args.variant}  components={list(seed_candidate.keys())}")

    adapter = CompoundProgramAdapter(
        actor=actor,
        score_fn=cfg["score"],
        extract_fn=cfg["extract"],
        full_scaffold=full_scaffold,
        max_workers=args.max_workers,
    )

    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"[rae] reflection_lm={args.reflection_lm}  budget={args.budget}  run_dir={args.run_dir}"
    )

    reflection_lm = _build_reflection_lm(args.reflection_lm, actor=actor)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=dev,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        module_selector=args.module_selector,
        reflection_minibatch_size=args.minibatch,
        max_metric_calls=args.budget,
        run_dir=args.run_dir,
        seed=args.seed,
        display_progress_bar=True,
        acceptance_criterion=args.acceptance,
    )

    best_dev = float(result.val_aggregate_scores[result.best_idx])
    print(
        f"[rae] best dev score: {best_dev:.4f}  (best_idx={result.best_idx} of {len(result.candidates)})"
    )
    best_cand = (
        result.best_candidate
        if isinstance(result.best_candidate, dict)
        else result.candidates[result.best_idx]
    )
    holdout_eval = adapter.evaluate(holdout, best_cand, capture_traces=False)
    holdout_acc = sum(holdout_eval.scores) / max(1, len(holdout_eval.scores))
    print(f"[rae] holdout accuracy: {holdout_acc:.4f} (n={len(holdout)})")

    summary = {
        "arena": args.arena,
        "variant": args.variant,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_holdout": len(holdout),
        "budget": args.budget,
        "reflection_lm": args.reflection_lm,
        "best_dev_score": best_dev,
        "holdout_accuracy": float(holdout_acc),
        "best_candidate": best_cand,
        "all_dev_scores": [float(s) for s in result.val_aggregate_scores],
    }
    out_path = Path(args.run_dir) / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[rae] wrote {out_path}")


if __name__ == "__main__":
    main()
