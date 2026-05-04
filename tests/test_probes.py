"""Unit tests for probe training logic."""

from __future__ import annotations

import numpy as np
import pytest

from rae.probes import (
    _bootstrap_auroc,
    _build_targets,
    _dom_probe_score,
    _eval_per_layer,
    _lr_l2_probe_score,
)


def test_build_targets_splits_correctly():
    records = [
        {"draft_correct": 1.0, "final_correct": 1.0},  # R->R
        {"draft_correct": 1.0, "final_correct": 0.0},  # R->W
        {"draft_correct": 0.0, "final_correct": 1.0},  # W->R
        {"draft_correct": 0.0, "final_correct": 0.0},  # W->W
        {"draft_correct": 0.0, "final_correct": 1.0},  # W->R
    ]
    t = _build_targets(records)

    mask_W, y_W = t["W->R"]
    mask_R, y_R = t["R->W"]
    _, y_C = t["correctness"]

    # 3 wrong drafts, 2 of which repaired.
    assert mask_W.sum() == 3
    assert y_W.sum() == 2

    # 2 correct drafts, 1 of which damaged.
    assert mask_R.sum() == 2
    assert y_R.sum() == 1

    # Correctness: 2 correct out of 5.
    assert y_C.sum() == 2


def test_dom_probe_finds_linear_direction():
    """On a noise-free linearly separable dataset, DoM AUROC should be 1.0."""
    rng = np.random.default_rng(0)
    # Direction v; class 1 is +v, class 0 is -v (+ noise).
    d = 32
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    X0 = rng.standard_normal((20, d)) - 2 * v
    X1 = rng.standard_normal((20, d)) + 2 * v
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(20), np.ones(20)]).astype(int)

    scores = _dom_probe_score(X, y, X)
    # Separable => scores rank-match labels.
    from sklearn.metrics import roc_auc_score

    auroc = roc_auc_score(y, scores)
    assert auroc > 0.95


def test_lr_l2_probe_trains():
    """LR+L2 probe should classify a trivial separable set perfectly."""
    rng = np.random.default_rng(1)
    d = 16
    X0 = rng.standard_normal((30, d)) - 3
    X1 = rng.standard_normal((30, d)) + 3
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(30), np.ones(30)]).astype(int)
    # Shuffle so train/test both see both classes.
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    scores = _lr_l2_probe_score(X[:40], y[:40], X[40:])
    from sklearn.metrics import roc_auc_score

    auroc = roc_auc_score(y[40:], scores)
    assert auroc > 0.9


def test_bootstrap_auroc_returns_valid_ci():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=100)
    scores = rng.standard_normal(100) + y  # weak signal
    base, lo, hi = _bootstrap_auroc(y, scores, n_boot=100, seed=7)
    assert 0.0 <= lo <= base <= hi <= 1.0


def test_eval_per_layer_returns_one_result_per_layer():
    rng = np.random.default_rng(3)
    n_layers = 5
    n = 40
    d = 16
    # Layer 3 has the signal; others are noise.
    H = rng.standard_normal((n, n_layers, d))
    y = rng.integers(0, 2, size=n)
    # Inject signal on layer 3.
    H[:, 3, :] += y[:, None] * 5.0

    results = _eval_per_layer(H, y, "test", "gen_last", "dom", n_splits=4, seed=0)

    assert len(results) == n_layers
    aurocs = [r.auroc for r in results]
    # Layer with signal should have highest AUROC.
    assert int(np.argmax(aurocs)) == 3
    assert max(aurocs) > 0.9
