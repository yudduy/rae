"""Advisability probes — conditional + unconditional.

Loads records.jsonl + per-problem npz hidden states, trains per-layer probes,
reports AUROC with problem-level bootstrap CIs.

Conditional targets (per council deliberation, 2026-04-20):

  Probe-W: P(W -> R | draft_wrong, h)
           Among wrong drafts, which will be repaired by forced advice?
  Probe-R: P(R -> W | draft_correct, h)
           Among correct drafts, which will be damaged by forced advice?

Unconditional baseline for comparison:

  Probe-C: P(draft_correct | h)   -- correctness probe (Cencerrado et al. 2509.10625)

Each probe is trained per-layer x position (prompt_last, gen_last), and a
best-layer is selected on a held-out problem fold to avoid cherry-picking.

Two probe families:
  - Difference-of-means (Cencerrado zero-free-parameter recipe): mu_pos - mu_neg.
  - L2 logistic regression (sklearn) -- same data, regularized.

CLI:
  python -m rae.probes --run-dir runs/advis_v1 --output-dir runs/advis_v1/probes
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


POSITIONS = ("prompt_last", "gen_last")


@dataclass
class ProbeResult:
    probe_name: str  # "W->R" | "R->W" | "correctness"
    position: str  # "prompt_last" | "gen_last"
    layer: int  # 0 = embedding, 1..N = block outputs
    method: str  # "dom" | "lr_l2"
    n_train: int
    n_test: int
    pos_rate: float  # fraction of positives in test
    auroc: float
    auroc_ci_lo: float
    auroc_ci_hi: float


def _load_dataset(run_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return records list, and stacked hidden-state arrays per position.

    Returns (records, H_prompt, H_gen) where records is list[dict] and
    H_prompt/H_gen have shape (n_problems, n_layers, hidden_size).
    """
    records: list[dict] = []
    with (run_dir / "records.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))

    # Keep only records with a hidden npz and no fatal error.
    kept = []
    H_prompt = []
    H_gen = []
    for r in records:
        path = Path(r["hidden_npz_path"])
        if r.get("error") and r.get("draft", "") == "":
            continue
        if not path.exists():
            continue
        with np.load(path) as npz:
            H_prompt.append(npz["prompt_last"])  # (n_layers, hidden)
            H_gen.append(npz["gen_last"])
        kept.append(r)
    if not kept:
        raise RuntimeError(f"No usable records in {run_dir}")
    return kept, np.stack(H_prompt, axis=0), np.stack(H_gen, axis=0)


def _dom_probe_score(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Difference-of-means probe: score = <x, mu_pos - mu_neg>."""
    pos = X_train[y_train == 1]
    neg = X_train[y_train == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(len(X_test))
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    return X_test @ direction


def _lr_l2_probe_score(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, C: float = 1.0
) -> np.ndarray:
    if len(set(y_train.tolist())) < 2:
        return np.zeros(len(X_test))
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def _bootstrap_auroc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Problem-level bootstrap CI over AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_true[idx], scores[idx]
        if len(set(yb.tolist())) < 2:
            continue
        aurocs.append(roc_auc_score(yb, sb))
    if not aurocs:
        return 0.5, 0.5, 0.5
    base = roc_auc_score(y_true, scores) if len(set(y_true.tolist())) > 1 else 0.5
    lo, hi = np.percentile(aurocs, [2.5, 97.5])
    return float(base), float(lo), float(hi)


def _eval_per_layer(
    H: np.ndarray,  # (n, n_layers, hidden)
    y: np.ndarray,
    probe_name: str,
    position: str,
    method: str,
    n_splits: int = 5,
    seed: int = 0,
) -> list[ProbeResult]:
    """5-fold stratified CV per layer; pool out-of-fold scores → AUROC."""
    n, n_layers, _ = H.shape
    results: list[ProbeResult] = []
    if len(set(y.tolist())) < 2:
        # Degenerate target: return zero results at 0.5 AUROC.
        return [
            ProbeResult(
                probe_name,
                position,
                layer=ell,
                method=method,
                n_train=0,
                n_test=n,
                pos_rate=float(y.mean()) if n else 0.0,
                auroc=0.5,
                auroc_ci_lo=0.5,
                auroc_ci_hi=0.5,
            )
            for ell in range(n_layers)
        ]
    skf = StratifiedKFold(
        n_splits=min(n_splits, int(y.sum()), int((y == 0).sum())),
        shuffle=True,
        random_state=seed,
    )
    score_fn = _dom_probe_score if method == "dom" else _lr_l2_probe_score
    for ell in range(n_layers):
        X = H[:, ell, :]
        all_scores = np.zeros(n)
        all_y = np.zeros(n)
        for tr, te in skf.split(X, y):
            all_scores[te] = score_fn(X[tr], y[tr], X[te])
            all_y[te] = y[te]
        auroc, lo, hi = _bootstrap_auroc(all_y, all_scores)
        results.append(
            ProbeResult(
                probe_name,
                position,
                layer=ell,
                method=method,
                n_train=n,
                n_test=n,
                pos_rate=float(y.mean()),
                auroc=auroc,
                auroc_ci_lo=lo,
                auroc_ci_hi=hi,
            )
        )
    return results


def _build_targets(records: list[dict]) -> dict[str, np.ndarray]:
    """Build the three target vectors + masks."""
    draft = np.array([float(r["draft_correct"]) for r in records])
    final = np.array([float(r["final_correct"]) for r in records])
    # Probe-W: among draft_wrong, 1 if W->R else 0.
    mask_W = draft == 0
    y_W = (final == 1)[mask_W].astype(int)
    # Probe-R: among draft_correct, 1 if R->W else 0.
    mask_R = draft == 1
    y_R = (final == 0)[mask_R].astype(int)
    # Probe-C: correctness.
    y_C = draft.astype(int)
    return {
        "W->R": (mask_W, y_W),
        "R->W": (mask_R, y_R),
        "correctness": (np.ones(len(records), dtype=bool), y_C),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=["dom", "lr_l2"])
    p.add_argument(
        "--permutation",
        action="store_true",
        help="Run label-shuffle permutation test on the best layer per config. "
        "Required when p ≫ n (bootstrap CI undercovers).",
    )
    p.add_argument("--n-permutations", type=int, default=500)
    args = p.parse_args()

    out_dir = args.output_dir or (args.run_dir / "probes")
    out_dir.mkdir(parents=True, exist_ok=True)

    records, H_prompt, H_gen = _load_dataset(args.run_dir)
    targets = _build_targets(records)

    print(f"[probes] n_problems={len(records)}")
    for name, (mask, y) in targets.items():
        n_pos = int(y.sum())
        n = int(mask.sum())
        print(
            f"[probes] target={name:12s}  n={n:4d}  pos={n_pos:4d}  rate={n_pos / max(n, 1):.3f}"
        )

    all_results: list[ProbeResult] = []
    for probe_name, (mask, y) in targets.items():
        if mask.sum() < 20:
            print(f"[probes] SKIP {probe_name}: only {mask.sum()} samples (<20)")
            continue
        for position, H_all in (("prompt_last", H_prompt), ("gen_last", H_gen)):
            H = H_all[mask]  # (n_filtered, n_layers, hidden)
            for method in args.methods:
                res = _eval_per_layer(
                    H, y, probe_name, position, method, n_splits=args.n_splits
                )
                all_results.extend(res)
                best = max(res, key=lambda r: r.auroc)
                print(
                    f"[probes] {probe_name:10s} {position:12s} {method:6s} "
                    f"best layer={best.layer:2d}  AUROC={best.auroc:.3f} "
                    f"[{best.auroc_ci_lo:.3f}, {best.auroc_ci_hi:.3f}]"
                )

    # Permutation test on best-layer per config (if requested).
    # When p >> n, bootstrap CIs undercover; label-shuffle permutation p-value is
    # the canonical non-parametric alternative. Shuffle labels within each
    # conditional mask, refit probe, record out-of-fold AUROC; p = fraction of
    # permuted AUROCs >= observed.
    perm_results: list[dict] = []
    if args.permutation:
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        # Index best-per-config by (probe, position, method).
        best_lookup: dict[tuple[str, str, str], ProbeResult] = {}
        for r in all_results:
            key = (r.probe_name, r.position, r.method)
            if key not in best_lookup or r.auroc > best_lookup[key].auroc:
                best_lookup[key] = r

        for key, best in best_lookup.items():
            probe_name, position, method = key
            if probe_name not in targets:
                continue
            mask, y = targets[probe_name]
            H_all = H_prompt if position == "prompt_last" else H_gen
            H = H_all[mask][:, best.layer, :]
            score_fn = _dom_probe_score if method == "dom" else _lr_l2_probe_score
            # Observed AUROC via same 5-fold CV as above.
            kf_splits = min(
                args.n_splits,
                int(y.sum()) if y.sum() > 0 else 1,
                int((y == 0).sum()) if (y == 0).sum() > 0 else 1,
            )
            if kf_splits < 2 or len(set(y.tolist())) < 2:
                continue
            skf = StratifiedKFold(n_splits=kf_splits, shuffle=True, random_state=0)
            oof = np.zeros(len(y))
            for tr, te in skf.split(H, y):
                oof[te] = score_fn(H[tr], y[tr], H[te])
            observed = roc_auc_score(y, oof)

            # Permutations.
            n_ge = 0
            for _ in range(args.n_permutations):
                y_perm = rng.permutation(y)
                oof_p = np.zeros(len(y_perm))
                for tr, te in skf.split(H, y_perm):
                    oof_p[te] = score_fn(H[tr], y_perm[tr], H[te])
                if len(set(y_perm[te].tolist())) < 2:
                    continue
                try:
                    a = roc_auc_score(y_perm, oof_p)
                except Exception:
                    continue
                if a >= observed:
                    n_ge += 1
            p_value = (n_ge + 1) / (args.n_permutations + 1)
            perm_results.append(
                {
                    "probe_name": probe_name,
                    "position": position,
                    "method": method,
                    "layer": best.layer,
                    "observed_auroc": float(observed),
                    "p_value": float(p_value),
                    "n_permutations": args.n_permutations,
                }
            )
            print(
                f"[perm] {probe_name:10s} {position:12s} {method:6s} L{best.layer} "
                f"observed={observed:.3f} p={p_value:.4f}"
            )

    # Save all results.
    with (out_dir / "results.jsonl").open("w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")
    if perm_results:
        with (out_dir / "permutations.jsonl").open("w") as f:
            for r in perm_results:
                f.write(json.dumps(r) + "\n")

    # Headline summary: best layer per (probe, position, method).
    headline: dict = {"n_problems": len(records), "best_per_config": []}
    for probe_name in targets:
        for position in POSITIONS:
            for method in args.methods:
                matching = [
                    r
                    for r in all_results
                    if r.probe_name == probe_name
                    and r.position == position
                    and r.method == method
                ]
                if not matching:
                    continue
                best = max(matching, key=lambda r: r.auroc)
                headline["best_per_config"].append(asdict(best))
    (out_dir / "summary.json").write_text(json.dumps(headline, indent=2))
    print(f"[probes] wrote {out_dir / 'results.jsonl'} and summary.json")


if __name__ == "__main__":
    main()
