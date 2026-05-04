"""K-draft cross-actor collection.

The decisive A-vs-B test prescribed by ChatGPT Pro + council deliberation
(2026-04-20):

  - For each problem, generate K independent drafts at temperature > 0.
  - For each draft, run full advisor scaffold with a DIFFERENT model as advisor
    (breaks self-correction symmetry).
  - Capture hidden states at the final draft token (per draft).
  - Record transition labels per draft.

A question-only probe must predict all K drafts of the same problem identically
(question features are invariant across drafts). An actor-state probe can
distinguish drafts of the same question that become R→W from drafts that
survive. This is the cleanest way to show that the probe reads something
beyond question difficulty.

Cross-actor design:
  actor_solve   → Qwen-7B via transformers (hidden states captured)
  diagnose      → Qwen-72B-AWQ via vLLM at localhost:8001
  advise        → Qwen-72B-AWQ via vLLM at localhost:8001
  actor_revise  → Qwen-7B via transformers (same frozen weights as solve)

Usage:
  python -m rae.collect_kdraft --run-dir runs/kdraft_v1 \\
      --n-problems 50 --k-drafts 4 --temperature 0.6 \\
      --data-src /workspace/rae/data_in/actor_eval_k64.jsonl \\
      --cache-dir /root/.cache/huggingface \\
      --advisor-base-url http://localhost:8001/v1 \\
      --advisor-api-key sk-capr-actor \\
      --advisor-model Qwen/Qwen2.5-72B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .actor_client import ActorClient, ActorConfig
from .arenas.math_zpd import MathProblem, load_problems
from .compound_program import is_no_advice
from .scorers import score_math
from .seed_prompts_math import forced_advise_seed_candidate_math


@dataclass
class KDraftRecord:
    problem_id: str
    draft_idx: int  # 0..K-1 for within-problem grouping
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
    transition: str
    delta_advise: float
    hidden_npz_path: str
    wall_seconds: float
    error: Optional[str] = None


def _classify(d: bool, f: bool) -> str:
    return f"{'R' if d else 'W'}->{'R' if f else 'W'}"


def _collect_one_draft(
    problem: MathProblem,
    draft_idx: int,
    actor_transformers,  # ActivationActor (Qwen-7B, captures activations)
    advisor_client: ActorClient,  # vLLM-backed 72B
    candidate: dict[str, str],
    hidden_dir: Path,
    *,
    temperature: float,
) -> KDraftRecord:
    import numpy as np

    t0 = time.time()
    err = None
    draft_text = diagnosis = advice_text = final_text = ""
    draft_score = final_score = 0.0
    advice_skipped = False
    hidden_path = hidden_dir / f"{problem.instance_id}_k{draft_idx}.npz"

    try:
        # 1. Draft with activation capture (actor, temperature > 0 for diversity).
        solve_msgs = [
            {"role": "system", "content": candidate["actor_solve"]},
            {"role": "user", "content": problem.question},
        ]
        # ActivationActor's config drives temperature; we pass kwargs downstream
        # via a temporary override on the underlying model generate call.
        # Cheapest path: mutate cfg.temperature for this call.
        orig_temp = actor_transformers.cfg.temperature
        actor_transformers.cfg = type(actor_transformers.cfg)(
            **{**actor_transformers.cfg.__dict__, "temperature": temperature}
        )
        try:
            draft_text, h = actor_transformers.chat_and_capture(solve_msgs)
        finally:
            actor_transformers.cfg = type(actor_transformers.cfg)(
                **{**actor_transformers.cfg.__dict__, "temperature": orig_temp}
            )
        draft_score = float(score_math(draft_text, problem.ground_truth))

        np.savez_compressed(
            hidden_path,
            prompt_last=h.prompt_last.astype("float32"),
            gen_last=h.gen_last.astype("float32"),
        )

        # 2. Advisor roles via the separate (72B) client. Temperature 0 for
        # deterministic advice given the draft.
        diag_msgs = [
            {"role": "system", "content": candidate["advisor_diagnose"]},
            {
                "role": "user",
                "content": f"PROBLEM:\n{problem.question}\n\nDRAFT:\n{draft_text}",
            },
        ]
        diagnosis = advisor_client.chat(diag_msgs, temperature=0.0)

        advise_msgs = [
            {"role": "system", "content": candidate["advisor_advise"]},
            {
                "role": "user",
                "content": (
                    f"PROBLEM:\n{problem.question}\n\nDRAFT:\n{draft_text}\n\nDIAGNOSIS:\n{diagnosis}"
                ),
            },
        ]
        advice_text = advisor_client.chat(advise_msgs, temperature=0.0)
        advice_skipped = is_no_advice(advice_text)

        # 3. Revise via actor (Qwen-7B) at temperature 0 (deterministic revision).
        revise_user = f"{advice_text}\n\n{candidate['actor_revise']}"
        revise_msgs = [
            {"role": "system", "content": candidate["actor_solve"]},
            {"role": "user", "content": problem.question},
            {"role": "assistant", "content": draft_text},
            {"role": "user", "content": revise_user},
        ]
        final_text = actor_transformers.chat(revise_msgs)
        final_score = float(score_math(final_text, problem.ground_truth))
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        if draft_text and not final_text:
            final_text = draft_text
            final_score = draft_score
            advice_skipped = True

    return KDraftRecord(
        problem_id=problem.instance_id,
        draft_idx=draft_idx,
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n-problems", type=int, default=50)
    p.add_argument("--k-drafts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--pass-lo", type=float, default=0.15)
    p.add_argument("--pass-hi", type=float, default=0.50)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--data-src", type=Path, default=None)
    p.add_argument("--advisor-base-url", default="http://localhost:8001/v1")
    p.add_argument("--advisor-api-key", default="sk-capr-actor")
    p.add_argument("--advisor-model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p.add_argument("--advisor-max-tokens", type=int, default=1024)
    args = p.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = args.run_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = dict(
        pass_lo=args.pass_lo, pass_hi=args.pass_hi, max_n=args.n_problems
    )
    if args.data_src is not None:
        load_kwargs["src"] = args.data_src
    problems = load_problems(**load_kwargs)
    print(
        f"[kdraft] loaded {len(problems)} problems (pass_rate in [{args.pass_lo}, {args.pass_hi}])"
    )

    from .activation_actor import ActivationActor, ActivationActorConfig

    actor = ActivationActor(
        ActivationActorConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            cache_dir=args.cache_dir,
            temperature=args.temperature,  # overridden per-call
        )
    )
    advisor = ActorClient(
        ActorConfig(
            base_url=args.advisor_base_url,
            api_key=args.advisor_api_key,
            model=args.advisor_model,
            max_tokens=args.advisor_max_tokens,
        )
    )

    candidate = forced_advise_seed_candidate_math()

    records_path = args.run_dir / "records.jsonl"
    records: list[KDraftRecord] = []
    t_start = time.time()
    n_total = len(problems) * args.k_drafts

    with records_path.open("w") as f:
        for i, prob in enumerate(problems):
            for k in range(args.k_drafts):
                rec = _collect_one_draft(
                    prob,
                    k,
                    actor,
                    advisor,
                    candidate,
                    hidden_dir,
                    temperature=args.temperature,
                )
                records.append(rec)
                f.write(json.dumps(asdict(rec)) + "\n")
                f.flush()
            # progress every problem (K draft records written)
            done = (i + 1) * args.k_drafts
            elapsed = time.time() - t_start
            dacc = sum(r.draft_correct for r in records) / len(records)
            facc = sum(r.final_correct for r in records) / len(records)
            wr = sum(1 for r in records if r.transition == "W->R")
            rw = sum(1 for r in records if r.transition == "R->W")
            print(
                f"[kdraft] {done}/{n_total} (prob {i + 1}/{len(problems)}) "
                f"draft_acc={dacc:.3f} final_acc={facc:.3f} "
                f"W->R={wr} R->W={rw} elapsed={elapsed:.0f}s"
            )

    # Summary with within-problem variance.
    wr_total = sum(1 for r in records if r.transition == "W->R")
    rw_total = sum(1 for r in records if r.transition == "R->W")
    rr_total = sum(1 for r in records if r.transition == "R->R")
    ww_total = sum(1 for r in records if r.transition == "W->W")

    # Per-problem: how many of K drafts have distinct outcomes?
    from collections import defaultdict

    prob_transitions = defaultdict(list)
    for r in records:
        prob_transitions[r.problem_id].append(r.transition)
    n_mixed = sum(1 for v in prob_transitions.values() if len(set(v)) > 1)

    summary = {
        "n_problems": len(problems),
        "k_drafts": args.k_drafts,
        "n_total_rollouts": len(records),
        "draft_acc": sum(r.draft_correct for r in records) / max(len(records), 1),
        "final_acc": sum(r.final_correct for r in records) / max(len(records), 1),
        "net_regularization": (wr_total - rw_total) / max(len(records), 1),
        "transitions": {
            "W->W": ww_total,
            "W->R": wr_total,
            "R->R": rr_total,
            "R->W": rw_total,
        },
        "mixed_outcome_problems": n_mixed,
        "mixed_rate": n_mixed / max(len(problems), 1),
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[kdraft] done. summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
