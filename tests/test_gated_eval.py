"""Unit tests for counterfactual gate simulation."""

from __future__ import annotations

import numpy as np

from rae.gated_eval import _counterfactual_accuracy


def test_always_advise_equals_final_acc():
    draft = np.array([0, 0, 1, 1, 0], dtype=float)
    final = np.array([1, 0, 1, 0, 0], dtype=float)
    decisions = np.ones_like(draft, dtype=int)
    acc, br = _counterfactual_accuracy(decisions, draft, final)
    assert acc == final.mean()
    assert br["advice_rate"] == 1.0
    assert br["n_WR"] == 1
    assert br["n_RW"] == 1
    assert br["n_RR"] == 1
    assert br["n_WW"] == 2


def test_never_advise_equals_draft_acc():
    draft = np.array([0, 0, 1, 1, 0], dtype=float)
    final = np.array([1, 0, 1, 0, 0], dtype=float)
    decisions = np.zeros_like(draft, dtype=int)
    acc, br = _counterfactual_accuracy(decisions, draft, final)
    assert acc == draft.mean()
    assert br["advice_rate"] == 0.0
    assert br["n_WR"] == 0
    assert br["n_RW"] == 0
    # Preserve: 2 correct -> R->R, 3 wrong -> W->W.
    assert br["n_RR"] == 2
    assert br["n_WW"] == 3


def test_oracle_dominates_always():
    """Oracle should beat always-advise when R->W exists."""
    draft = np.array([0, 0, 1, 1, 0], dtype=float)
    final = np.array([1, 0, 1, 0, 0], dtype=float)
    always_dec = np.ones_like(draft, dtype=int)
    # Oracle: advise iff final > draft.
    oracle_dec = (final > draft).astype(int)
    acc_always, _ = _counterfactual_accuracy(always_dec, draft, final)
    acc_oracle, _ = _counterfactual_accuracy(oracle_dec, draft, final)
    # Always-advise loses the R->W case; oracle preserves both R's.
    assert acc_oracle > acc_always
    assert acc_oracle == 3 / 5  # 1 W->R + 2 R->R preserved = 3 right


def test_net_reg_matches_definition():
    """Net regularization = (W->R - R->W) / n among advise-decisions."""
    draft = np.array([0, 0, 1, 1, 0], dtype=float)
    final = np.array([1, 0, 1, 0, 0], dtype=float)
    decisions = np.ones_like(draft, dtype=int)
    _, br = _counterfactual_accuracy(decisions, draft, final)
    # (1 W->R - 1 R->W) / 5 = 0
    assert br["net_reg"] == 0.0
