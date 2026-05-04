"""Paired rollout + activation capture driver.

For each MATH ZPD problem (pass_rate ∈ [0.15, 0.50]):

  1. actor_solve(question) → draft, residual_stream@end_of_draft   (transformers)
  2. score draft  → draft_correct ∈ {0, 1}
  3. advisor_diagnose + advisor_advise(draft)  → advice | NO_ADVICE
  4. actor_revise(draft, advice)  → final       (skipped on NO_ADVICE)
  5. score final  → final_correct ∈ {0, 1}

Preserve-branch final = draft (no call needed). Advisability per problem:
  Δ_advise = final_correct - draft_correct  ∈ {-1, 0, +1}

Outputs (run_dir):
  records.jsonl        — one row per problem: {id, draft, diag, advice, final,
                         draft_correct, final_correct, advice_skipped,
                         transition, delta_advise, hidden_npz_path}
  hidden/{id}.npz      — {prompt_last, gen_last}  shape (n_layers, hidden_size)
  summary.json         — counts, rates, class balance

No GPU => unit tests stub out `ActivationActor`; driver logic is pure Python.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .arenas.math_zpd import MathProblem, load_problems
from .compound_program import is_no_advice, run_advise, run_diagnose, run_revise
from .scorers import score_math
from .seed_prompts_math import (
    default_seed_candidate_math,
    forced_advise_seed_candidate_math,
)


@dataclass
class AdvisabilityRecord:
    instance_id: str
    level: str
    subject: str
    pass_rate_baseline: float
    question: str
    ground_truth: str
    draft: str
    diagnosis: str
    advice: str
    final: str
    draft_correct: float
    final_correct: float
    advice_skipped: bool
    transition: str  # R->R, R->W, W->R, W->W (no _NA suffix; we include advice_skipped)
    delta_advise: float
    hidden_npz_path: str
    wall_seconds: float
    error: Optional[str] = None


def _classify(draft_ok: bool, final_ok: bool) -> str:
    d = "R" if draft_ok else "W"
    f = "R" if final_ok else "W"
    return f"{d}->{f}"


def collect_one(
    problem: MathProblem,
    actor,
    candidate: dict[str, str],
    hidden_dir: Path,
    *,
    force_revise: bool = False,
) -> AdvisabilityRecord:
    """Run one paired rollout and capture activations."""
    import numpy as np

    t0 = time.time()
    err = None
    draft_text = diagnosis = advice_text = final_text = ""
    advice_skipped = False
    draft_score = final_score = 0.0
    hidden_path = hidden_dir / f"{problem.instance_id}.npz"

    try:
        # 1. Draft with activation capture.
        solve_msgs = [
            {"role": "system", "content": candidate["actor_solve"]},
            {"role": "user", "content": problem.question},
        ]
        draft_text, h = actor.chat_and_capture(solve_msgs)
        draft_score = float(score_math(draft_text, problem.ground_truth))

        # Persist hidden states now so large npz don't sit in RAM across the run.
        np.savez_compressed(
            hidden_path,
            prompt_last=h.prompt_last.astype("float32"),
            gen_last=h.gen_last.astype("float32"),
        )

        # 2. Full scaffold (diagnose → advise → revise).
        diagnosis = run_diagnose(candidate, problem.question, draft_text, actor)
        advice_text = run_advise(
            candidate, problem.question, draft_text, diagnosis, actor
        )

        if is_no_advice(advice_text) and not force_revise:
            advice_skipped = True
            final_text = draft_text
            final_score = draft_score
        else:
            advice_skipped_flag = is_no_advice(
                advice_text
            )  # still revise if force_revise
            advice_skipped = advice_skipped_flag and not force_revise
            final_text = run_revise(
                candidate,
                problem.question,
                draft_text,
                advice_text,
                actor,
                actor_solve_system=candidate["actor_solve"],
            )
            final_score = float(score_math(final_text, problem.ground_truth))
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        # Fallback: if draft exists, treat as preserve branch.
        if draft_text and not final_text:
            final_text = draft_text
            final_score = draft_score
            advice_skipped = True

    return AdvisabilityRecord(
        instance_id=problem.instance_id,
        level=problem.level,
        subject=problem.subject,
        pass_rate_baseline=problem.pass_rate_baseline,
        question=problem.question,
        ground_truth=problem.ground_truth,
        draft=draft_text,
        diagnosis=diagnosis,
        advice=advice_text,
        final=final_text,
        draft_correct=draft_score,
        final_correct=final_score,
        advice_skipped=advice_skipped,
        transition=_classify(bool(draft_score), bool(final_score)),
        delta_advise=final_score - draft_score,
        hidden_npz_path=str(hidden_path),
        wall_seconds=time.time() - t0,
        error=err,
    )


def _summary(records: list[AdvisabilityRecord]) -> dict:
    n = len(records)
    n_ok = sum(1 for r in records if r.error is None)
    counter: dict[str, int] = {}
    advice_emitted = no_advice = 0
    for r in records:
        counter[r.transition] = counter.get(r.transition, 0) + 1
        if r.advice_skipped:
            no_advice += 1
        else:
            advice_emitted += 1
    n_wr = sum(1 for r in records if r.transition == "W->R")
    n_rw = sum(1 for r in records if r.transition == "R->W")
    return {
        "n": n,
        "n_ok": n_ok,
        "draft_acc": sum(r.draft_correct for r in records) / max(n, 1),
        "final_acc": sum(r.final_correct for r in records) / max(n, 1),
        "net_regularization": (n_wr - n_rw) / max(n, 1),
        "advice_rate": advice_emitted / max(n, 1),
        "transitions": counter,
        "advice_emitted": advice_emitted,
        "no_advice": no_advice,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--max-n", type=int, default=200)
    p.add_argument("--pass-lo", type=float, default=0.15)
    p.add_argument("--pass-hi", type=float, default=0.50)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--data-src",
        type=Path,
        default=None,
        help="Override path to actor_eval_k64.jsonl (default: hardcoded local path)",
    )
    p.add_argument(
        "--force-advise",
        action="store_true",
        help="Use the forced-advise seed (advisor never emits NO_ADVICE). "
        "Required for unbiased Δ_advise measurement; abstention is trained "
        "as a downstream gate on the probe.",
    )
    p.add_argument(
        "--force-revise",
        action="store_true",
        help="Always run revise even if advisor emits NO_ADVICE. Unconditional "
        "paired-rollout mode.",
    )
    args = p.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = args.run_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = dict(pass_lo=args.pass_lo, pass_hi=args.pass_hi, max_n=args.max_n)
    if args.data_src is not None:
        load_kwargs["src"] = args.data_src
    problems = load_problems(**load_kwargs)
    print(
        f"[collect] loaded {len(problems)} problems (pass_rate in [{args.pass_lo}, {args.pass_hi}])"
    )

    from .activation_actor import ActivationActor, ActivationActorConfig

    actor = ActivationActor(
        ActivationActorConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            cache_dir=args.cache_dir,
        )
    )

    candidate = (
        forced_advise_seed_candidate_math()
        if args.force_advise
        else default_seed_candidate_math()
    )

    records_path = args.run_dir / "records.jsonl"
    records: list[AdvisabilityRecord] = []
    t_start = time.time()
    with records_path.open("w") as f:
        for i, prob in enumerate(problems):
            rec = collect_one(
                prob, actor, candidate, hidden_dir, force_revise=args.force_revise
            )
            records.append(rec)
            f.write(json.dumps(asdict(rec)) + "\n")
            f.flush()
            if (i + 1) % 5 == 0 or i + 1 == len(problems):
                elapsed = time.time() - t_start
                s = _summary(records)
                print(
                    f"[collect] {i + 1}/{len(problems)} "
                    f"draft_acc={s['draft_acc']:.3f} final_acc={s['final_acc']:.3f} "
                    f"net_reg={s['net_regularization']:+.3f} "
                    f"advice_rate={s['advice_rate']:.3f} "
                    f"elapsed={elapsed:.0f}s  "
                    f"err={rec.error or 'ok'}"
                )

    summary = _summary(records)
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[collect] done. summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
