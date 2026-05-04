"""Seed prompts for the MATH ZPD arena.

The actor must end with `\\boxed{<answer>}`. The advisor diagnoses common
math failure modes (algebra error, missed case, geometric assumption, etc.)
and emits a concrete corrective hint without solving.
"""

from __future__ import annotations

ACTOR_SOLVE_SEED_MATH = (
    "Solve the math problem step by step. Show your reasoning. End your "
    "response with the final answer in \\boxed{} format."
)

ADVISOR_DIAGNOSE_SEED_MATH = (
    "You are a senior math reviewer reading a student's draft solution. Read "
    "the problem and the draft, then identify the single most likely failure "
    "mode. Choose from: (a) algebra/arithmetic slip, (b) missed a case or "
    "constraint, (c) wrong formula or theorem applied, (d) sign or domain "
    "error, (e) misread the problem statement, (f) final answer not in "
    "\\boxed{} format. Output exactly two short lines:\n"
    "FAILURE_MODE: <letter>\n"
    "EVIDENCE: <one sentence pointing to the specific step in the draft>"
)

ADVISOR_ADVISE_SEED_MATH = (
    "You are an advisor. Given the problem, the draft solution, and the "
    "diagnosis above, write ONE concrete corrective hint (<= 2 sentences) "
    "that points the student to the specific step or rule to re-check. Do "
    "NOT solve the problem. Do NOT restate the answer. If the diagnosis is "
    "weak or the draft already looks correct, output exactly: NO_ADVICE."
)


# Forced version for advisability-probe data collection: never abstain, always
# emit a concrete hint so the Δ = final_with_advice - draft signal is measured,
# not masked by the advisor's abstention policy. The abstention decision is a
# separate question trained as a downstream gate on top of the probe.
ADVISOR_ADVISE_SEED_MATH_FORCED = (
    "You are an advisor. Given the problem, the draft solution, and the "
    "diagnosis above, write ONE concrete corrective hint (<= 2 sentences) "
    "that points the student to the specific step, rule, or sanity check "
    "most worth re-examining. Do NOT solve the problem. Do NOT restate the "
    "answer. Always emit a hint; NEVER output NO_ADVICE. If you are unsure, "
    "suggest a verification step (e.g., plug the answer back in, recount "
    "cases, or double-check arithmetic on a specific line)."
)

ACTOR_REVISE_SEED_MATH = (
    "Revise your previous solution ONLY if the advisor's hint identifies a "
    "concrete issue. Preserve any correct prior work. End your response with "
    "the final answer in \\boxed{} format."
)


def default_seed_candidate_math() -> dict[str, str]:
    return {
        "actor_solve": ACTOR_SOLVE_SEED_MATH,
        "advisor_diagnose": ADVISOR_DIAGNOSE_SEED_MATH,
        "advisor_advise": ADVISOR_ADVISE_SEED_MATH,
        "actor_revise": ACTOR_REVISE_SEED_MATH,
    }


def forced_advise_seed_candidate_math() -> dict[str, str]:
    """Seed used by advisability-probe data collection: always emits advice."""
    return {
        "actor_solve": ACTOR_SOLVE_SEED_MATH,
        "advisor_diagnose": ADVISOR_DIAGNOSE_SEED_MATH,
        "advisor_advise": ADVISOR_ADVISE_SEED_MATH_FORCED,
        "actor_revise": ACTOR_REVISE_SEED_MATH,
    }


def actor_only_seed_candidate_math() -> dict[str, str]:
    return {"actor_solve": ACTOR_SOLVE_SEED_MATH}
