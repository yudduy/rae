"""Unit tests for collect_advisability driver logic (no GPU required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from rae.arenas.math_zpd import MathProblem
from rae.collect_advisability import (
    AdvisabilityRecord,
    _classify,
    _summary,
    collect_one,
)
from rae.seed_prompts_math import default_seed_candidate_math


def _make_problem(ans: str = "42") -> MathProblem:
    return MathProblem(
        instance_id="test_001",
        level="Level 3",
        subject="Algebra",
        pass_rate_baseline=0.3,
        question="What is the answer?",
        ground_truth=ans,
    )


class FakeActor:
    """Mimics ActivationActor for driver-logic tests. Returns fake hidden states."""

    def __init__(self, draft: str, diagnosis: str, advice: str, revised: str):
        self.draft = draft
        self.diagnosis = diagnosis
        self.advice = advice
        self.revised = revised
        self.call_log: list[str] = []

    def chat_and_capture(self, messages):
        self.call_log.append("solve")
        import numpy as np

        h = SimpleNamespace(
            prompt_last=np.zeros((29, 8), dtype="float32"),  # 28 layers + embedding
            gen_last=np.ones((29, 8), dtype="float32"),
        )
        return self.draft, h

    def chat(self, messages, **kw):
        # Which role is this? Inspect the last user message.
        user_text = messages[-1]["content"]
        sys_text = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "DIAGNOSIS:" in user_text:
            self.call_log.append("advise")
            return self.advice
        if "DRAFT:" in user_text and "DIAGNOSIS:" not in user_text:
            self.call_log.append("diagnose")
            return self.diagnosis
        # revise: 4-message conversation with assistant turn
        if any(m["role"] == "assistant" for m in messages):
            self.call_log.append("revise")
            return self.revised
        return ""


def test_classify_matrix():
    assert _classify(True, True) == "R->R"
    assert _classify(True, False) == "R->W"
    assert _classify(False, True) == "W->R"
    assert _classify(False, False) == "W->W"


def test_collect_one_wr_repair(tmp_path: Path):
    """Wrong draft → advisor emits hint → correct final. W->R, Δ=+1."""
    prob = _make_problem("42")
    actor = FakeActor(
        draft="The answer is \\boxed{7}.",  # wrong
        diagnosis="FAILURE_MODE: a\nEVIDENCE: arithmetic",
        advice="Recompute.",
        revised="The answer is \\boxed{42}.",  # correct
    )
    rec = collect_one(prob, actor, default_seed_candidate_math(), tmp_path)

    assert rec.draft_correct == 0.0
    assert rec.final_correct == 1.0
    assert rec.transition == "W->R"
    assert rec.delta_advise == 1.0
    assert rec.advice_skipped is False
    assert rec.error is None
    assert "solve" in actor.call_log
    assert "revise" in actor.call_log
    assert Path(rec.hidden_npz_path).exists()


def test_collect_one_rr_preservation(tmp_path: Path):
    """Correct draft → advisor emits NO_ADVICE → final = draft. R->R, Δ=0."""
    prob = _make_problem("42")
    actor = FakeActor(
        draft="The answer is \\boxed{42}.",
        diagnosis="FAILURE_MODE: f\nEVIDENCE: looks ok",
        advice="NO_ADVICE",
        revised="should not be called",
    )
    rec = collect_one(prob, actor, default_seed_candidate_math(), tmp_path)

    assert rec.draft_correct == 1.0
    assert rec.final_correct == 1.0
    assert rec.transition == "R->R"
    assert rec.delta_advise == 0.0
    assert rec.advice_skipped is True
    assert "revise" not in actor.call_log


def test_collect_one_rw_over_advising(tmp_path: Path):
    """Correct draft → advisor emits advice → revised is wrong. R->W, Δ=-1 (over-advising)."""
    prob = _make_problem("42")
    actor = FakeActor(
        draft="The answer is \\boxed{42}.",
        diagnosis="FAILURE_MODE: b\nEVIDENCE: recheck",
        advice="Reconsider the sign.",
        revised="The answer is \\boxed{-42}.",
    )
    rec = collect_one(prob, actor, default_seed_candidate_math(), tmp_path)

    assert rec.draft_correct == 1.0
    assert rec.final_correct == 0.0
    assert rec.transition == "R->W"
    assert rec.delta_advise == -1.0
    assert rec.advice_skipped is False


def test_summary_rates(tmp_path: Path):
    """Summary computes transition counts + net regularization correctly."""
    records = [
        AdvisabilityRecord(
            instance_id=f"id_{i}",
            level="",
            subject="",
            pass_rate_baseline=0.3,
            question="",
            ground_truth="",
            draft="",
            diagnosis="",
            advice="",
            final="",
            draft_correct=dc,
            final_correct=fc,
            advice_skipped=skip,
            transition=_classify(bool(dc), bool(fc)),
            delta_advise=fc - dc,
            hidden_npz_path="",
            wall_seconds=0.0,
            error=None,
        )
        for i, (dc, fc, skip) in enumerate(
            [
                (0.0, 1.0, False),  # W->R
                (0.0, 1.0, False),  # W->R
                (1.0, 0.0, False),  # R->W (over-advise)
                (1.0, 1.0, True),  # R->R preserve
                (0.0, 0.0, False),  # W->W
            ]
        )
    ]
    s = _summary(records)
    assert s["n"] == 5
    assert s["draft_acc"] == 2 / 5
    assert s["final_acc"] == 3 / 5
    # Net reg = (W->R - R->W)/n = (2-1)/5 = 0.2
    assert abs(s["net_regularization"] - 0.2) < 1e-9
    assert s["advice_emitted"] == 4
    assert s["no_advice"] == 1
    assert s["transitions"]["W->R"] == 2
    assert s["transitions"]["R->W"] == 1


def test_collect_one_error_recovers_to_preserve(tmp_path: Path):
    """If an advisor step raises, driver falls back to preserve branch."""
    prob = _make_problem("42")
    actor = FakeActor(
        draft="The answer is \\boxed{7}.",
        diagnosis="",
        advice="",
        revised="",
    )

    # Force run_diagnose to raise by overriding FakeActor.chat to fail on first call.
    orig_chat = actor.chat

    def failing_chat(messages, **kw):
        raise RuntimeError("synthetic failure")

    actor.chat = failing_chat

    rec = collect_one(prob, actor, default_seed_candidate_math(), tmp_path)

    # Draft still scored; final falls back to draft.
    assert rec.draft_correct == 0.0
    assert rec.final_correct == 0.0
    assert rec.advice_skipped is True
    assert rec.error is not None and "synthetic failure" in rec.error
