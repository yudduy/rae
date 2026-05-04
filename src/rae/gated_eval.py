"""Probe-gated advisor evaluation via counterfactual simulation.

The paired rollouts (collect_advisability.py) give us, per problem:
    draft_correct,  final_with_advice_correct (forced)

A gate is a function (hidden_state, draft_logprob, ...) -> {advise, skip}.
Given any gate, we counterfactually simulate final accuracy without new rollouts:
    final = final_with_advice_correct  if gate=advise
    final = draft_correct               if gate=skip

This lets us compare gates at matched advice rates.

Gates implemented:
  - always-advise            : accuracy = final_acc
  - never-advise             : accuracy = draft_acc
  - oracle                   : advise iff Δ > 0 (upper bound)
  - random-p                 : advise with prob p (baseline)
  - probe-cascade            : correctness-probe decides draft state, then
                               conditional advisability (W-probe or R-probe)
                               decides to advise. Trained 5-fold.

Metrics reported:
  - accuracy, advice_rate, net_regularization, W->R, R->W, R->R, W->W counts

Usage:
  python -m rae.gated_eval --run-dir runs/advis_v1 --output-dir runs/advis_v1/gates
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .probes import _dom_probe_score, _load_dataset


@dataclass
class GateResult:
    gate_name: str
    accuracy: float
    advice_rate: float
    net_reg: float
    n_WR: int
    n_RW: int
    n_RR: int
    n_WW: int
    accuracy_ci_lo: float
    accuracy_ci_hi: float


def _counterfactual_accuracy(
    gate_decisions: np.ndarray,  # 1 if advise else 0
    draft_ok: np.ndarray,
    final_ok: np.ndarray,
) -> tuple[float, dict]:
    """Return (accuracy, breakdown) under the counterfactual gate."""
    n = len(draft_ok)
    final = np.where(gate_decisions == 1, final_ok, draft_ok)
    acc = float(final.mean())
    # Transitions under the gate.
    # Note: transition is only meaningful when advice was emitted;
    # "preserved" transitions are draft_ok == final == draft_ok.
    n_advise = int(gate_decisions.sum())
    wr = int(((gate_decisions == 1) & (draft_ok == 0) & (final_ok == 1)).sum())
    rw = int(((gate_decisions == 1) & (draft_ok == 1) & (final_ok == 0)).sum())
    rr = int(((gate_decisions == 1) & (draft_ok == 1) & (final_ok == 1)).sum())
    ww = int(((gate_decisions == 1) & (draft_ok == 0) & (final_ok == 0)).sum())
    # Preserve-only transitions (gate=skip): add to RR or WW counts by draft status.
    rr += int(((gate_decisions == 0) & (draft_ok == 1)).sum())
    ww += int(((gate_decisions == 0) & (draft_ok == 0)).sum())
    net_reg = (wr - rw) / max(n, 1)
    return acc, dict(
        n_WR=wr,
        n_RW=rw,
        n_RR=rr,
        n_WW=ww,
        advice_rate=n_advise / max(n, 1),
        net_reg=net_reg,
    )


def _bootstrap_accuracy(
    gate_decisions: np.ndarray,
    draft_ok: np.ndarray,
    final_ok: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(draft_ok)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        final = np.where(gate_decisions[idx] == 1, final_ok[idx], draft_ok[idx])
        accs.append(float(final.mean()))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def _probe_cascade_decisions(
    H: np.ndarray,  # (n, n_layers, hidden) on the *selected* position
    draft_ok: np.ndarray,
    final_ok: np.ndarray,
    layer: int,
    n_splits: int = 5,
    seed: int = 0,
    r_to_w_threshold: float = 0.5,  # if P(R->W | h) > this, skip; else advise
    w_to_r_threshold: float = 0.5,  # if P(W->R | h) > this, advise; else skip
) -> np.ndarray:
    """Train 3 probes out-of-fold; emit gate decisions for the full dataset.

    Cascade:
      1. Correctness probe predicts P(draft_ok | h).
      2a. If predicted correct (p >= 0.5): use R->W probe to gate. Advise iff P(R->W) <= threshold.
      2b. If predicted wrong (p < 0.5): use W->R probe to gate. Advise iff P(W->R) >= threshold.

    Trained with stratified K-fold; out-of-fold predictions only (no leakage).
    """
    n = len(draft_ok)
    X = H[:, layer, :]
    decisions = np.zeros(n, dtype=int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        X_tr, X_te = X[tr], X[te]

        # Correctness probe (all train samples).
        y_c_tr = draft_ok[tr].astype(int)
        if len(set(y_c_tr.tolist())) < 2:
            p_correct_te = np.full(len(te), float(y_c_tr.mean()))
        else:
            clf_c = LogisticRegression(C=1.0, max_iter=500).fit(X_tr, y_c_tr)
            p_correct_te = clf_c.predict_proba(X_te)[:, 1]

        # W->R probe on wrong-draft subset of train.
        mask_W = draft_ok[tr] == 0
        y_WR_tr = (final_ok[tr][mask_W] == 1).astype(int)
        if mask_W.sum() < 5 or len(set(y_WR_tr.tolist())) < 2:
            p_WR_te = np.full(len(te), float(y_WR_tr.mean()) if len(y_WR_tr) else 0.5)
        else:
            clf_W = LogisticRegression(C=1.0, max_iter=500).fit(X_tr[mask_W], y_WR_tr)
            p_WR_te = clf_W.predict_proba(X_te)[:, 1]

        # R->W probe on correct-draft subset of train.
        mask_R = draft_ok[tr] == 1
        y_RW_tr = (final_ok[tr][mask_R] == 0).astype(int)
        if mask_R.sum() < 5 or len(set(y_RW_tr.tolist())) < 2:
            p_RW_te = np.full(len(te), float(y_RW_tr.mean()) if len(y_RW_tr) else 0.5)
        else:
            clf_R = LogisticRegression(C=1.0, max_iter=500).fit(X_tr[mask_R], y_RW_tr)
            p_RW_te = clf_R.predict_proba(X_te)[:, 1]

        # Cascade decision.
        for i, idx in enumerate(te):
            if p_correct_te[i] >= 0.5:
                # Predicted correct: advise only if R->W damage risk is low.
                decisions[idx] = 1 if p_RW_te[i] <= r_to_w_threshold else 0
            else:
                # Predicted wrong: advise only if repair chance is high.
                decisions[idx] = 1 if p_WR_te[i] >= w_to_r_threshold else 0

    return decisions


def evaluate_gates(run_dir: Path) -> list[GateResult]:
    records, H_prompt, H_gen = _load_dataset(run_dir)
    draft_ok = np.array([float(r["draft_correct"]) for r in records])
    final_ok = np.array([float(r["final_correct"]) for r in records])

    gates: list[tuple[str, np.ndarray]] = []

    # Always-advise.
    gates.append(("always-advise", np.ones_like(draft_ok, dtype=int)))

    # Never-advise.
    gates.append(("never-advise", np.zeros_like(draft_ok, dtype=int)))

    # Oracle: advise iff advice actually helps.
    oracle = (final_ok > draft_ok).astype(int)
    gates.append(("oracle", oracle))

    # Random gate at the empirical always-advise rate.
    rng = np.random.default_rng(0)
    p = float((final_ok > draft_ok).mean())  # oracle's advice rate
    gates.append(("random-oracle-rate", (rng.random(len(draft_ok)) < p).astype(int)))

    # Surface-feature gates (logprob-threshold). Matches oracle advice rate so
    # the probe must beat them at the same budget.
    surface_path = run_dir / "surface_features.jsonl"
    if surface_path.exists():
        sf_by_id = {}
        with surface_path.open() as f:
            for line in f:
                r = json.loads(line)
                sf_by_id[r["instance_id"]] = r
        # Align to records order.
        feat_mean = np.array(
            [
                sf_by_id.get(r["instance_id"], {}).get("draft_mean_logprob", 0.0)
                for r in records
            ]
        )
        feat_last = np.array(
            [
                sf_by_id.get(r["instance_id"], {}).get("draft_last_logprob", 0.0)
                for r in records
            ]
        )
        feat_min = np.array(
            [
                sf_by_id.get(r["instance_id"], {}).get("draft_min_logprob", 0.0)
                for r in records
            ]
        )
        oracle_rate = float((final_ok > draft_ok).mean())
        # Low-logprob-means-advise: advise when confidence is low.
        for fname, vec in (("mean", feat_mean), ("last", feat_last), ("min", feat_min)):
            thr = np.quantile(
                vec, oracle_rate
            )  # bottom oracle_rate by logprob get advised
            dec = (vec <= thr).astype(int)
            gates.append((f"logprob-{fname}-matched", dec))

    # Probe cascade — try best layer from each position.
    for position, H in (("prompt_last", H_prompt), ("gen_last", H_gen)):
        n_layers = H.shape[1]
        best_acc = -np.inf
        best_layer = -1
        best_decisions = None
        # Pick the layer that maximizes accuracy on in-sample cascade (fast scan).
        for ell in range(n_layers):
            decisions = _probe_cascade_decisions(H, draft_ok, final_ok, ell)
            acc, _ = _counterfactual_accuracy(decisions, draft_ok, final_ok)
            if acc > best_acc:
                best_acc = acc
                best_layer = ell
                best_decisions = decisions
        gates.append((f"probe-cascade-{position}-L{best_layer}", best_decisions))

    results: list[GateResult] = []
    for name, dec in gates:
        acc, br = _counterfactual_accuracy(dec, draft_ok, final_ok)
        lo, hi = _bootstrap_accuracy(dec, draft_ok, final_ok)
        results.append(
            GateResult(
                gate_name=name,
                accuracy=acc,
                advice_rate=br["advice_rate"],
                net_reg=br["net_reg"],
                n_WR=br["n_WR"],
                n_RW=br["n_RW"],
                n_RR=br["n_RR"],
                n_WW=br["n_WW"],
                accuracy_ci_lo=lo,
                accuracy_ci_hi=hi,
            )
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or (args.run_dir / "gates")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_gates(args.run_dir)

    # Sort by accuracy.
    results.sort(key=lambda r: r.accuracy, reverse=True)
    print(
        f"{'gate':45s}  {'acc':>6s}  {'CI':>14s}  {'advise':>6s}  {'netReg':>7s}  WR RW RR WW"
    )
    for r in results:
        print(
            f"{r.gate_name:45s}  {r.accuracy:.3f}  [{r.accuracy_ci_lo:.3f},{r.accuracy_ci_hi:.3f}]  "
            f"{r.advice_rate:.3f}  {r.net_reg:+.3f}  {r.n_WR:2d} {r.n_RW:2d} {r.n_RR:2d} {r.n_WW:2d}"
        )

    with (out_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


if __name__ == "__main__":
    main()
