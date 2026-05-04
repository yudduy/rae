"""Compound program shape with a mocked Actor.

We don't hit vLLM here; we replace ActorClient.chat with a script that returns
canned responses keyed by the system prompt. This validates: control flow,
chat-message construction, NO_ADVICE short-circuit, and the advisor-models
3-step revision layout (system, user-question, assistant-draft, user-advice).
"""

from typing import Optional

from rae.actor_client import ActorClient
from rae.compound_program import (
    NO_ADVICE_SENTINEL,
    is_no_advice,
    run_compound,
    run_revise,
)
from rae.seed_prompts import default_seed_candidate


class FakeActor(ActorClient):
    """ActorClient that returns scripted responses keyed by substring match.

    Match priority: scan the LAST user message first (this is where revise's
    actor_revise instruction lives -- per advisor-models 3-step layout the
    revise call re-uses the actor_solve system prompt and puts the revise
    instruction in a trailing user turn). Then fall back to system prompt.
    """

    def __init__(self, scripts: dict[str, str]):
        self.scripts = scripts
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, temperature=None, max_tokens=None, seed=None):
        self.calls.append(messages)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        for key, response in self.scripts.items():
            if key in last_user:
                return response
        for key, response in self.scripts.items():
            if key in sys_prompt:
                return response
        return ""


def test_no_advice_sentinel_short_circuits_revision():
    cand = default_seed_candidate()
    actor = FakeActor(
        {
            cand["actor_solve"][:30]: "DRAFT_TEXT The total tax owed is $50.",
            cand["advisor_diagnose"][:30]: "FAILURE_MODE: a\nEVIDENCE: x",
            cand["advisor_advise"][:30]: NO_ADVICE_SENTINEL,
        }
    )
    trace = run_compound(cand, "Q?", actor, full_scaffold=True)
    assert trace.draft.startswith("DRAFT_TEXT")
    assert trace.advice_skipped is True
    assert trace.final == trace.draft
    # Exactly 3 calls: solve, diagnose, advise. No revise.
    assert len(actor.calls) == 3


def test_actor_only_variant_does_not_call_advisor_modules():
    cand = {"actor_solve": "You are a helpful US taxation consultant."}
    actor = FakeActor({"taxation": "DRAFT The total tax owed is $5."})
    trace = run_compound(cand, "Compute this.", actor, full_scaffold=False)
    assert trace.advice_skipped is True
    assert trace.final == trace.draft
    assert len(actor.calls) == 1


def test_full_scaffold_calls_all_four_modules_when_advice_emitted():
    cand = default_seed_candidate()
    actor = FakeActor(
        {
            cand["actor_solve"][:30]: "DRAFT The total tax owed is $50.",
            cand["advisor_diagnose"][:30]: "FAILURE_MODE: e\nEVIDENCE: arithmetic slip",
            cand["advisor_advise"][:30]: "Re-check the addition on line 3.",
            cand["actor_revise"][:30]: "REVISED The total tax owed is $52.",
        }
    )
    trace = run_compound(cand, "Q?", actor, full_scaffold=True)
    assert len(actor.calls) == 4
    assert trace.advice_skipped is False
    assert trace.final.startswith("REVISED")


def test_revise_message_layout_matches_advisor_models_3step():
    """advisor_models/rule_arena/env.py _build_student_prompt sends:
       [system=actor_solve] [user=question] [assistant=draft] [user=advice + revise].
    Verify our run_revise emits the same 4-message structure in the same order.
    """
    cand = default_seed_candidate()
    actor = FakeActor({"taxation": "REVISED The total tax owed is $7."})
    run_revise(
        cand, "Q?", "DRAFT", "advice", actor, actor_solve_system=cand["actor_solve"]
    )
    msgs = actor.calls[-1]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == cand["actor_solve"]
    assert msgs[1]["content"] == "Q?"
    assert msgs[2]["content"] == "DRAFT"
    assert "advice" in msgs[3]["content"]
    assert cand["actor_revise"] in msgs[3]["content"]


def test_is_no_advice_recognizes_variants():
    assert is_no_advice("NO_ADVICE")
    assert is_no_advice("NO_ADVICE.")
    assert is_no_advice("no_advice ")
    assert is_no_advice("")
    assert is_no_advice(None)  # type: ignore[arg-type]
    assert not is_no_advice("Re-check line 5.")


def test_compound_handles_actor_exception_gracefully():
    cand = default_seed_candidate()

    class BoomActor(ActorClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, *, temperature=None, max_tokens=None, seed=None):
            self.calls += 1
            raise RuntimeError("vLLM down")

    trace = run_compound(cand, "Q?", BoomActor(), full_scaffold=True)
    assert trace.error is not None
    assert "vLLM down" in trace.error
