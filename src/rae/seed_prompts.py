"""Seed candidate prompts for the 4-module compound program.

These are the initial theta values GEPA mutates. Module names are stable keys
in the candidate dict consumed by `gepa.optimize_anything`.

Bootstrapped verbatim where possible from advisor-models config.py
(STUDENT_SYSTEM_PROMPT, ADVISOR_INSTRUCTIONS) so we measure the GEPA delta
against the published Advisor Models 2-step setup, not against arbitrary text.
"""

from __future__ import annotations

# Verbatim STUDENT_SYSTEM_PROMPT from advisor_models/rule_arena/config.py
ACTOR_SOLVE_SEED = (
    "You are a helpful US taxation consultant. End your response with: '1. The "
    "total tax owed is $xxx.' (xxx is a number) if there is tax owed. 2. The "
    "total tax overpaid is $xxx.' (xxx is a number) if there is tax overpaid "
    "(and should be refunded)."
)

# A "thinking" step the advisor does INTERNALLY before emitting advice.
# This is novel relative to published Advisor Models (which collapses
# diagnose+advise into one ADVISOR_INSTRUCTIONS turn).
ADVISOR_DIAGNOSE_SEED = (
    "You are a senior US tax reviewer reading a junior consultant's draft "
    "answer. Read the original problem and the draft below, then identify the "
    "single most likely failure mode. Choose from: (a) wrong rule applied, "
    "(b) right rule but wrong number, (c) missed an exception or threshold, "
    "(d) representation error (units, sign, owed-vs-overpaid), (e) arithmetic, "
    "(f) format / final-line missing. Output two short lines:\n"
    "FAILURE_MODE: <letter>\n"
    "EVIDENCE: <one sentence pointing to the specific step in the draft>"
)

# Inspired by ADVISOR_INSTRUCTIONS but specialised: advice not solution.
ADVISOR_ADVISE_SEED = (
    "You are an advisor. Given the problem, the draft answer, and the "
    "diagnosis above, write ONE concrete corrective hint (<= 2 sentences) that "
    "tells the consultant exactly which step to re-check or which rule to apply. "
    "Do NOT solve the problem. Do NOT restate the answer. If the diagnosis is "
    "weak or the draft already looks correct, output exactly: NO_ADVICE."
)

# Revision instruction. Receives original question, original draft (assistant
# turn), and advisor advice (user turn) per advisor-models 3-step pattern.
ACTOR_REVISE_SEED = (
    "Revise your previous answer ONLY if the advisor's hint identifies a "
    "concrete issue. Preserve any correct prior work. End your response with "
    "the same final line format as before: '1. The total tax owed is $xxx.' "
    "or '2. The total tax overpaid is $xxx.'"
)


def default_seed_candidate() -> dict[str, str]:
    return {
        "actor_solve": ACTOR_SOLVE_SEED,
        "advisor_diagnose": ADVISOR_DIAGNOSE_SEED,
        "advisor_advise": ADVISOR_ADVISE_SEED,
        "actor_revise": ACTOR_REVISE_SEED,
    }


def actor_only_seed_candidate() -> dict[str, str]:
    """Experiment 1 baseline: GEPA evolves only the actor system prompt."""
    return {"actor_solve": ACTOR_SOLVE_SEED}
