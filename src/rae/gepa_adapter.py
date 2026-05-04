"""GEPAAdapter implementation for the RuleArena Taxes compound program.

This is the integration point between GEPA's reflective evolution engine
(gepa.optimize) and our 4-module Actor-Advisor compound program. Two methods:

  1. evaluate(batch, candidate, capture_traces) -> EvaluationBatch
     Run the compound program on each TaxProblem in the batch (parallel over
     problems, sequential within a single rollout). Score each. If
     capture_traces=True, save per-example trajectories for later reflection.

  2. make_reflective_dataset(candidate, eval_batch, components_to_update)
     Build the small JSON-serializable dataset GEPA's teacher LM consumes.
     Schema: {component_name: [{"Inputs", "Generated Outputs", "Feedback"}]}.

Schema rationale: per the GEPAAdapter docstring, the "Feedback" field is the
single highest-leverage knob -- it is the rich textual signal the reflection
LM mutates against. We engineer per-module feedback that calls out exactly
which failure mode occurred (over-advising, missed format, regression, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from .actor_client import ActorClient
from .compound_program import Trace, run_compound

# Output type stored per example. Lightweight & JSON-serializable.
RolloutOutput = dict

# Score function: (response_text, ground_truth) -> {0.0, 1.0}
ScoreFn = Callable[[str, Any], float]
# Optional extractor: (response_text) -> something printable for feedback
ExtractFn = Callable[[str], Any]


class _ProblemLike(Protocol):
    instance_id: str
    question: str
    ground_truth: Any


class CompoundProgramAdapter(GEPAAdapter[_ProblemLike, Trace, RolloutOutput]):
    """Arena-agnostic GEPAAdapter for the 4-module compound program.

    Parametrised by `score_fn` (regex+np.isclose for Tax, math_verify for MATH)
    and an optional `extract_fn` (used to render the predicted value in the
    reflective Feedback strings -- floats for Tax, boxed expressions for MATH).

    Thread-safety: ActorClient wraps OpenAI() which is thread-safe; we use a
    process-wide ThreadPoolExecutor sized to `max_workers`.
    """

    def __init__(
        self,
        actor: ActorClient,
        score_fn: ScoreFn,
        *,
        extract_fn: ExtractFn | None = None,
        full_scaffold: bool = True,
        max_workers: int = 8,
    ):
        self.actor = actor
        self.score_fn = score_fn
        self.extract_fn = extract_fn or (lambda s: None)
        self.full_scaffold = full_scaffold
        self.max_workers = max_workers

    # ------------------------------------------------------------------ evaluate
    def evaluate(
        self,
        batch: list[_ProblemLike],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trace, RolloutOutput]:
        def run_one(p: _ProblemLike) -> tuple[float, RolloutOutput, Trace]:
            trace = run_compound(
                candidate, p.question, self.actor, full_scaffold=self.full_scaffold
            )
            score = self.score_fn(trace.final, p.ground_truth)
            draft_score = self.score_fn(trace.draft, p.ground_truth)
            output: RolloutOutput = {
                "instance_id": p.instance_id,
                "ground_truth": p.ground_truth,
                "extracted_final": self.extract_fn(trace.final),
                "extracted_draft": self.extract_fn(trace.draft),
                "final_score": float(score),
                "draft_score": float(draft_score),
                "advice_skipped": trace.advice_skipped,
                "error": trace.error,
            }
            return float(score), output, trace

        outputs: list[RolloutOutput] = [None] * len(batch)  # type: ignore
        scores: list[float] = [0.0] * len(batch)
        trajectories: list[Trace] | None = (
            [None] * len(batch) if capture_traces else None
        )  # type: ignore

        if not batch:
            return EvaluationBatch(
                outputs=[], scores=[], trajectories=[] if capture_traces else None
            )

        # Parallel rollout. Each compound program call is several sequential
        # LLM hops; parallelism is across problems, not within a single rollout.
        max_workers = min(self.max_workers, len(batch))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_one, p): i for i, p in enumerate(batch)}
            for fut, i in futures.items():
                try:
                    score, out, tr = fut.result()
                except Exception as e:  # noqa: BLE001
                    # Per adapter contract: fall back to score 0 + error trace,
                    # never raise on individual example failures.
                    p = batch[i]
                    out = {
                        "instance_id": p.instance_id,
                        "ground_truth": p.ground_truth,
                        "final_score": 0.0,
                        "error": f"adapter exception: {e}",
                    }
                    tr = Trace(question=p.question, error=f"adapter exception: {e}")
                    score = 0.0
                scores[i] = score
                outputs[i] = out
                if trajectories is not None:
                    trajectories[i] = tr

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories
        )

    # ----------------------------------------------------- make_reflective_dataset
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trace, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        out: dict[str, list[dict]] = {c: [] for c in components_to_update}
        traces = eval_batch.trajectories or []
        outputs = eval_batch.outputs or []
        scores = eval_batch.scores or []

        for tr, raw_out, score in zip(traces, outputs, scores):
            if tr is None:
                continue
            gt = raw_out.get("ground_truth") if isinstance(raw_out, dict) else None
            draft_score = (raw_out or {}).get("draft_score", 0.0)
            final_score = (raw_out or {}).get("final_score", 0.0)
            extracted_final = (raw_out or {}).get("extracted_final")
            extracted_draft = (raw_out or {}).get("extracted_draft")

            for comp in components_to_update:
                rec = self._record_for_component(
                    comp,
                    candidate=candidate,
                    trace=tr,
                    ground_truth=gt,
                    draft_score=draft_score,
                    final_score=final_score,
                    extracted_final=extracted_final,
                    extracted_draft=extracted_draft,
                )
                if rec is not None:
                    out[comp].append(rec)
        return out

    # ---- per-component reflective records ----
    def _record_for_component(
        self,
        comp: str,
        *,
        candidate: dict[str, str],
        trace: Trace,
        ground_truth: Any,
        draft_score: float,
        final_score: float,
        extracted_final: Any,
        extracted_draft: Any,
    ) -> dict | None:
        question = _truncate(trace.question, 2200)
        if comp == "actor_solve":
            feedback = self._feedback_actor_solve(
                trace, ground_truth, draft_score, extracted_draft
            )
            return {
                "Inputs": {"problem": question},
                "Generated Outputs": {"draft_answer": _truncate(trace.draft, 1200)},
                "Feedback": feedback,
            }
        if comp == "advisor_diagnose":
            feedback = self._feedback_advisor_diagnose(trace, draft_score)
            return {
                "Inputs": {
                    "problem": question,
                    "draft": _truncate(trace.draft, 800),
                },
                "Generated Outputs": {"diagnosis": _truncate(trace.diagnosis, 800)},
                "Feedback": feedback,
            }
        if comp == "advisor_advise":
            feedback = self._feedback_advisor_advise(trace, draft_score, final_score)
            return {
                "Inputs": {
                    "problem": question,
                    "draft": _truncate(trace.draft, 800),
                    "diagnosis": _truncate(trace.diagnosis, 600),
                },
                "Generated Outputs": {"advice": _truncate(trace.advice, 800)},
                "Feedback": feedback,
            }
        if comp == "actor_revise":
            feedback = self._feedback_actor_revise(
                trace, draft_score, final_score, extracted_final, ground_truth
            )
            return {
                "Inputs": {
                    "problem": question,
                    "original_draft": _truncate(trace.draft, 800),
                    "advice": _truncate(trace.advice, 600),
                },
                "Generated Outputs": {"final_answer": _truncate(trace.final, 1200)},
                "Feedback": feedback,
            }
        return None

    @staticmethod
    def _feedback_actor_solve(
        trace: Trace, gt: Any, draft_score: float, extracted_draft: Any
    ) -> str:
        if draft_score == 1.0:
            return (
                f"Draft is CORRECT (extracted={extracted_draft}, gold={gt}). The advisor "
                "step will be at risk of breaking it -- the draft system prompt is doing "
                "its job here; do NOT regress."
            )
        if extracted_draft is None:
            return (
                f"Draft FAILED extraction. Gold answer={gt}. The required final-line "
                "format ('The total tax owed is $X.' or 'overpaid') is missing or "
                "mis-formatted. Strengthen the format instruction in the system prompt."
            )
        return (
            f"Draft is WRONG: extracted={extracted_draft}, gold={gt}. "
            "Identify which class of error this is (rule mis-application, arithmetic, "
            "missed exception, sign/representation) and reshape the system prompt to "
            "preempt it on similar cases."
        )

    @staticmethod
    def _feedback_advisor_diagnose(trace: Trace, draft_score: float) -> str:
        if not trace.diagnosis:
            return (
                "Diagnosis was empty -- the diagnose prompt did not elicit a response."
            )
        up = trace.diagnosis.upper()
        struct_ok = ("FAILURE_MODE" in up) and ("EVIDENCE" in up)
        parts = []
        if not struct_ok:
            parts.append(
                "Output did NOT follow the required FAILURE_MODE / EVIDENCE schema. "
                "The advise step depends on this structure -- fix the diagnose prompt to enforce it."
            )
        if draft_score == 1.0:
            parts.append(
                "Note: the draft was already correct. A robust diagnose prompt should "
                "be willing to declare 'no failure' when the draft is sound."
            )
        else:
            parts.append(
                "Draft was wrong; diagnose must surface a SPECIFIC step or rule, not a generic critique."
            )
        return " ".join(parts)

    @staticmethod
    def _feedback_advisor_advise(
        trace: Trace, draft_score: float, final_score: float
    ) -> str:
        if trace.advice_skipped:
            if draft_score == 1.0:
                return (
                    "Advice correctly suppressed (NO_ADVICE) on an already-correct "
                    "draft. This is the desired over-advising-safe behavior."
                )
            return (
                "Advice was suppressed (NO_ADVICE) but the draft was WRONG. The "
                "advise step failed to find leverage. Encourage emitting concrete "
                "hints when the diagnosis identifies a real failure."
            )
        if draft_score == 1.0 and final_score == 0.0:
            return (
                "OVER-ADVISING REGRESSION: advice was emitted on a correct draft, "
                "and revision broke it. Tighten suppression: only emit advice when "
                "diagnosis cites concrete evidence; otherwise emit NO_ADVICE."
            )
        if draft_score == 0.0 and final_score == 1.0:
            return "Advice REPAIRED a wrong draft. Preserve and reinforce this pattern."
        if draft_score == 0.0 and final_score == 0.0:
            return (
                "Advice was emitted but did not help. Either advice was too vague, "
                "or it solved the wrong subproblem. Make hints more specific to the "
                "concrete step the diagnosis pointed to. Do NOT include the answer."
            )
        return "Draft was already correct and revision preserved it. Acceptable."

    @staticmethod
    def _feedback_actor_revise(
        trace: Trace,
        draft_score: float,
        final_score: float,
        extracted_final: Any,
        gt: Any,
    ) -> str:
        if trace.advice_skipped:
            return "Revision step not taken (advice suppressed)."
        if draft_score == 1.0 and final_score == 0.0:
            return (
                f"REGRESSION: draft was correct, revision produced wrong final answer "
                f"(extracted={extracted_final}, gold={gt}). The revise prompt is over-"
                f"reacting to advice. Make it skeptical: only revise when the hint "
                f"identifies a concrete, unambiguous error."
            )
        if draft_score == 0.0 and final_score == 1.0:
            return "Revision REPAIRED the draft. Strong signal -- preserve."
        if draft_score == 0.0 and final_score == 0.0:
            return (
                f"Revision did not repair (still wrong: extracted={extracted_final}, "
                f"gold={gt}). The revise prompt may be discarding the advice or "
                f"mis-applying it. Improve integration of the corrective hint."
            )
        return "Final correct; revision behaved well."


def _truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + " ...[truncated]"
