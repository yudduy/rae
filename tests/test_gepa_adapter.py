"""GEPA adapter shape: evaluate() and make_reflective_dataset() contracts."""

from rae.actor_client import ActorClient
from rae.arenas.rule_arena_taxes import TaxProblem
from rae.compound_program import NO_ADVICE_SENTINEL
from rae.evaluator import extract_signed_amount, score_response
from rae.gepa_adapter import CompoundProgramAdapter
from rae.seed_prompts import default_seed_candidate


class FakeActor(ActorClient):
    """Same matcher pattern as tests/test_compound_program.py FakeActor:
    last user message wins (so revise call routes via its actor_revise key
    in the trailing user turn, not via the actor_solve system prompt)."""

    def __init__(self, scripts: dict[str, str]):
        self.scripts = scripts

    def chat(self, messages, *, temperature=None, max_tokens=None, seed=None):
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        for key, resp in self.scripts.items():
            if key in last_user:
                return resp
        for key, resp in self.scripts.items():
            if key in sys_prompt:
                return resp
        return ""


def _problem(idx: int, gold: float) -> TaxProblem:
    return TaxProblem(
        instance_id=f"t{idx}",
        complexity=0,
        question=f"Compute taxes for case #{idx}.",
        ground_truth=gold,
        info_dict={},
    )


def test_evaluate_scores_match_groundtruth_with_correct_drafts():
    cand = default_seed_candidate()
    actor = FakeActor(
        {
            cand["actor_solve"][:30]: "DRAFT The total tax owed is $42.",
            cand["advisor_diagnose"][:30]: "FAILURE_MODE: a\nEVIDENCE: looks fine",
            cand["advisor_advise"][:30]: NO_ADVICE_SENTINEL,  # suppress -> draft kept
        }
    )
    adapter = CompoundProgramAdapter(
        actor=actor,
        score_fn=score_response,
        extract_fn=extract_signed_amount,
        full_scaffold=True,
        max_workers=2,
    )
    batch = [_problem(0, 42.0), _problem(1, 100.0)]
    eb = adapter.evaluate(batch, cand, capture_traces=True)
    assert len(eb.scores) == 2
    assert eb.scores[0] == 1.0  # 42 matches
    assert eb.scores[1] == 0.0  # draft says 42, gold is 100
    assert eb.trajectories is not None
    assert eb.outputs[0]["instance_id"] == "t0"


def test_make_reflective_dataset_emits_per_component_records():
    cand = default_seed_candidate()
    actor = FakeActor(
        {
            cand["actor_solve"][:30]: "DRAFT The total tax owed is $5.",
            cand["advisor_diagnose"][:30]: "FAILURE_MODE: e\nEVIDENCE: arithmetic",
            cand["advisor_advise"][:30]: "Re-check the sum on line 4.",
            cand["actor_revise"][:30]: "REVISED The total tax owed is $7.",
        }
    )
    adapter = CompoundProgramAdapter(
        actor=actor,
        score_fn=score_response,
        extract_fn=extract_signed_amount,
        full_scaffold=True,
        max_workers=1,
    )
    batch = [_problem(0, 7.0)]  # revision lands on 7 -> correct
    eb = adapter.evaluate(batch, cand, capture_traces=True)
    rd = adapter.make_reflective_dataset(
        cand, eb, components_to_update=["actor_solve", "advisor_advise", "actor_revise"]
    )
    assert set(rd.keys()) == {"actor_solve", "advisor_advise", "actor_revise"}
    for comp, recs in rd.items():
        assert len(recs) == 1
        assert "Inputs" in recs[0]
        assert "Generated Outputs" in recs[0]
        assert "Feedback" in recs[0]
        assert isinstance(recs[0]["Feedback"], str)


def test_reflective_feedback_flags_over_advising_regression():
    """Draft was right; advisor emitted advice; revision broke it -> Feedback must call this out."""
    cand = default_seed_candidate()
    actor = FakeActor(
        {
            cand["actor_solve"][:30]: "DRAFT The total tax owed is $42.",
            cand["advisor_diagnose"][:30]: "FAILURE_MODE: e\nEVIDENCE: maybe",
            cand["advisor_advise"][:30]: "Reconsider the standard deduction.",
            cand["actor_revise"][:30]: "REVISED The total tax owed is $99.",
        }
    )
    adapter = CompoundProgramAdapter(
        actor=actor,
        score_fn=score_response,
        extract_fn=extract_signed_amount,
        full_scaffold=True,
        max_workers=1,
    )
    eb = adapter.evaluate([_problem(0, 42.0)], cand, capture_traces=True)
    rd = adapter.make_reflective_dataset(
        cand, eb, components_to_update=["advisor_advise", "actor_revise"]
    )
    advise_fb = rd["advisor_advise"][0]["Feedback"]
    revise_fb = rd["actor_revise"][0]["Feedback"]
    assert "OVER-ADVISING" in advise_fb
    assert "REGRESSION" in revise_fb


def test_evaluate_does_not_raise_on_actor_failure():
    cand = default_seed_candidate()

    class BoomActor(ActorClient):
        def __init__(self):
            pass

        def chat(self, *a, **kw):
            raise RuntimeError("nope")

    adapter = CompoundProgramAdapter(
        actor=BoomActor(),
        score_fn=score_response,
        extract_fn=extract_signed_amount,
        full_scaffold=True,
        max_workers=1,
    )
    eb = adapter.evaluate([_problem(0, 1.0)], cand, capture_traces=True)
    assert eb.scores == [0.0]
    assert (
        eb.outputs[0].get("error") is not None or eb.trajectories[0].error is not None
    )  # type: ignore
