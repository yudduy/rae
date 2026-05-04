"""4-module compound textual program.

    actor_solve  -->  draft_answer
    advisor_diagnose(question, draft)  -->  failure_hypothesis
    advisor_advise(question, draft, diagnosis)  -->  advice  (or NO_ADVICE)
    actor_revise(question, draft, advice)  -->  final_answer

The candidate dict GEPA evolves has keys: actor_solve, advisor_diagnose,
advisor_advise, actor_revise. The chat structure for revision deliberately
mirrors advisor_models/rule_arena/env.py _build_student_prompt: the original
draft becomes an assistant turn, then advice is the next user turn -- this
preserves the published Advisor Models conditioning so the GEPA delta is
attributable to scaffold evolution, not to a different chat layout.

If `advice == NO_ADVICE` (or empty) we skip revision and return the draft as
final, matching the over-advising-aware pattern from the user's design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .actor_client import ActorClient
from .seed_prompts import default_seed_candidate

NO_ADVICE_SENTINEL = "NO_ADVICE"


@dataclass
class Trace:
    question: str
    draft: str = ""
    diagnosis: str = ""
    advice: str = ""
    final: str = ""
    advice_skipped: bool = False
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "draft": self.draft,
            "diagnosis": self.diagnosis,
            "advice": self.advice,
            "final": self.final,
            "advice_skipped": self.advice_skipped,
            "error": self.error,
        }


def _need_keys(candidate: dict[str, str], required: tuple[str, ...]) -> None:
    missing = [k for k in required if k not in candidate]
    if missing:
        raise KeyError(f"candidate missing required modules: {missing}")


def run_solve(candidate: dict[str, str], question: str, actor: ActorClient) -> str:
    _need_keys(candidate, ("actor_solve",))
    msgs = [
        {"role": "system", "content": candidate["actor_solve"]},
        {"role": "user", "content": question},
    ]
    return actor.chat(msgs)


def run_diagnose(
    candidate: dict[str, str], question: str, draft: str, actor: ActorClient
) -> str:
    _need_keys(candidate, ("advisor_diagnose",))
    msgs = [
        {"role": "system", "content": candidate["advisor_diagnose"]},
        {"role": "user", "content": f"PROBLEM:\n{question}\n\nDRAFT:\n{draft}"},
    ]
    return actor.chat(msgs)


def run_advise(
    candidate: dict[str, str],
    question: str,
    draft: str,
    diagnosis: str,
    actor: ActorClient,
) -> str:
    _need_keys(candidate, ("advisor_advise",))
    msgs = [
        {"role": "system", "content": candidate["advisor_advise"]},
        {
            "role": "user",
            "content": (
                f"PROBLEM:\n{question}\n\nDRAFT:\n{draft}\n\nDIAGNOSIS:\n{diagnosis}"
            ),
        },
    ]
    return actor.chat(msgs)


def run_revise(
    candidate: dict[str, str],
    question: str,
    draft: str,
    advice: str,
    actor: ActorClient,
    actor_solve_system: str,
) -> str:
    """Revision turn matches advisor_models 3-step layout:
    [system=actor_solve] [user=question] [assistant=draft] [user=advice + revise instruction]."""
    _need_keys(candidate, ("actor_revise",))
    revise_user = f"{advice}\n\n{candidate['actor_revise']}"
    msgs = [
        {"role": "system", "content": actor_solve_system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": draft},
        {"role": "user", "content": revise_user},
    ]
    return actor.chat(msgs)


def is_no_advice(advice: str) -> bool:
    if not advice:
        return True
    head = advice.strip().splitlines()[0].strip().rstrip(".:")
    return head.upper().startswith(NO_ADVICE_SENTINEL)


def run_compound(
    candidate: dict[str, str],
    question: str,
    actor: ActorClient,
    *,
    full_scaffold: bool = True,
) -> Trace:
    """Run the 4-module compound program.

    full_scaffold=False -> Experiment 1 actor-only baseline (skip advisor).
    """
    trace = Trace(question=question)
    try:
        trace.draft = run_solve(candidate, question, actor)

        if not full_scaffold or "advisor_diagnose" not in candidate:
            trace.final = trace.draft
            trace.advice_skipped = True
            return trace

        trace.diagnosis = run_diagnose(candidate, question, trace.draft, actor)
        trace.advice = run_advise(
            candidate, question, trace.draft, trace.diagnosis, actor
        )

        if is_no_advice(trace.advice):
            trace.final = trace.draft
            trace.advice_skipped = True
            return trace

        trace.final = run_revise(
            candidate,
            question,
            trace.draft,
            trace.advice,
            actor,
            actor_solve_system=candidate["actor_solve"],
        )
    except Exception as e:  # noqa: BLE001
        trace.error = str(e)
        if not trace.final:
            trace.final = trace.draft
    return trace
