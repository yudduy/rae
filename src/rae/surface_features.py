"""Surface-feature baselines for gate comparison.

The advisability-probe thesis only survives if hidden-state probes beat
cheap surface features. This module computes, per problem:

  - draft_mean_logprob:   mean per-token log-likelihood of the draft
                          under the frozen actor (teacher-forced)
  - draft_last_logprob:   log-likelihood of the final draft token
  - draft_min_logprob:    min per-token log-likelihood in the draft
  - draft_perplexity:     exp(-mean_logprob)
  - draft_len_tokens:     token count of the draft

These are computed POST-HOC from records.jsonl so the main collection run
doesn't need to change. A separate teacher-forced forward pass over
[question || draft] gives per-token log-probs -- we only need the tokens
generated (not the prompt).

Usage:
  python -m rae.surface_features --run-dir runs/advis_v1 \\
      --model-id Qwen/Qwen2.5-7B-Instruct --cache-dir /root/.cache/huggingface
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SurfaceFeatures:
    instance_id: str
    draft_mean_logprob: float
    draft_last_logprob: float
    draft_min_logprob: float
    draft_perplexity: float
    draft_len_tokens: int


def _teacher_forced_logprobs(tokenizer, model, question: str, draft: str, device: str):
    """Return per-token log-probs for the draft portion only."""
    import torch

    # Build the same prompt used at generation time: user-turn question via
    # chat template with add_generation_prompt=True, then append the draft.
    msgs = [{"role": "user", "content": question}]
    prefix_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
    draft_ids = tokenizer(
        draft, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    full_ids = torch.cat([prefix_ids, draft_ids], dim=1)

    with torch.inference_mode():
        out = model(full_ids)
    logits = out.logits[0]  # (seq_len, vocab)
    # log-prob of token at position i is based on logits at position i-1.
    log_probs_full = torch.log_softmax(logits, dim=-1)
    # Tokens we care about: draft tokens only (positions [prefix_len, prefix_len+draft_len-1]).
    prefix_len = prefix_ids.shape[1]
    draft_len = draft_ids.shape[1]
    if draft_len == 0:
        return []
    draft_token_logprobs = []
    for i in range(draft_len):
        target_idx = prefix_len + i
        target_token = full_ids[0, target_idx].item()
        lp = log_probs_full[target_idx - 1, target_token].item()
        draft_token_logprobs.append(lp)
    return draft_token_logprobs


def compute_features_for_record(
    record: dict, tokenizer, model, device: str
) -> SurfaceFeatures | None:
    draft = record.get("draft") or ""
    question = record.get("question") or ""
    if not draft.strip():
        return None
    logprobs = _teacher_forced_logprobs(tokenizer, model, question, draft, device)
    if not logprobs:
        return None
    n = len(logprobs)
    mean_lp = sum(logprobs) / n
    last_lp = logprobs[-1]
    min_lp = min(logprobs)
    import math

    perp = math.exp(-mean_lp)
    return SurfaceFeatures(
        instance_id=record["instance_id"],
        draft_mean_logprob=mean_lp,
        draft_last_logprob=last_lp,
        draft_min_logprob=min_lp,
        draft_perplexity=perp,
        draft_len_tokens=n,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    records = []
    with (args.run_dir / "records.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[surface] loaded {len(records)} records from {args.run_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kwargs = {}
    mdl_kwargs = {
        "torch_dtype": getattr(torch, args.dtype),
        "device_map": "cuda",
    }
    if args.cache_dir:
        tok_kwargs["cache_dir"] = args.cache_dir
        mdl_kwargs["cache_dir"] = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **mdl_kwargs)
    model.eval()

    out_path = args.run_dir / "surface_features.jsonl"
    n_ok = 0
    n_err = 0
    with out_path.open("w") as f:
        for i, rec in enumerate(records):
            try:
                feat = compute_features_for_record(rec, tokenizer, model, "cuda")
                if feat is None:
                    n_err += 1
                    continue
                f.write(json.dumps(asdict(feat)) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[surface] ERR {rec.get('instance_id')}: {e}")
                n_err += 1
            if (i + 1) % 25 == 0:
                print(f"[surface] {i + 1}/{len(records)}  ok={n_ok}  err={n_err}")

    print(f"[surface] done. ok={n_ok} err={n_err}  wrote {out_path}")


if __name__ == "__main__":
    main()
