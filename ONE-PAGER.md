# Textual Strategy Compilation for Black-Box LLM Advisors

*Reflective Advisor Evolution: compiling control policies into context.*

## Thesis

A frozen LLM is not a fixed capability number. It is a conditional dynamical
system whose trajectory distribution is selected by context. The scientific
question is not "does prompting help" but:

> **How much of a learned control policy for a frozen executor can be
> compiled into text rather than into weights?**

Two artefacts from Berkeley Sky Lab instantiate both answers to the same
abstract problem:

- **GEPA** (Agrawal et al. 2025, arXiv:2507.19457) — reflective prompt
  evolution. Compiles control policies into **text**.
- **Advisor Models** (Asawa et al. 2025, arXiv:2510.02453) — small advisor
  trained with GRPO. Compiles control policies into **weights**.

They are two compilation targets for one abstract object: a control policy
over a frozen executor. Sky Lab has not published the comparison — their
`baselines/gepa/gepa_rule_arena.py` optimizes a single-module actor prompt
with no advisor in the loop.

## Define the compilation gap

Let π\_weight = RL-trained advisor policy (Asawa), π\_text = GEPA-evolved
compound scaffold (this work). Define:

$$
G_\text{compile} = J(\pi_\text{weight}) - J(\pi_\text{text})
$$

measured on the same actor, same data split, same inference harness.

- $G_\text{compile} \approx 0$ ⟹ strong claim: **text is a sufficient
  representational substrate for advisor policies** over this regime. Training
  is one route; evolution is another; they converge.
- $G_\text{compile} > 0$ ⟹ characterizes the **limits of context as a control
  medium** — i.e. the class of advisor behaviors that require amortization
  into parameters. Also a publishable finding.

## The compound program (object of optimization)

Per instance, four sequential LLM calls on the same frozen actor, with four
prompts θ₁..θ₄ that GEPA mutates:

```
θ₁ actor_solve       (problem)                          → draft
θ₂ advisor_diagnose  (problem, draft)                   → FAILURE_MODE + EVIDENCE
θ₃ advisor_advise    (problem, draft, diagnosis)        → hint  |  NO_ADVICE
θ₄ actor_revise      (problem, draft, hint)             → final  (skipped on NO_ADVICE)
```

Revise turn mirrors Asawa's 3-step chat layout verbatim, so our numbers are
directly comparable to their published baselines.

## The deeper objects

The scaffold gives us three separable phenomena to study, independent of the
accuracy headline:

### 1. Trajectory regularization (empirical signature)

Capability lives in a trajectory distribution, not a mean. Define the
transition table per problem:

| Draft | Final | Name |
|-------|-------|------|
| wrong | right | **repair** |
| right | right | **preservation** |
| right | wrong | **over-advising regression** |
| wrong | wrong | failed repair |
| right | NO_ADVICE → right | successful abstention |
| wrong | NO_ADVICE → wrong | missed opportunity |

Headline metrics the paper should track alongside accuracy:

$$
\text{NetReg} = P(W \to R) - P(R \to W), \quad
\text{OverAdvise} = P(R \to W \mid \text{advice emitted})
$$

A good advisor scaffold is not maximally talkative. It is selectively causal
— it must learn **silence** on correct drafts.

### 2. Role multiplexing (mechanism)

The compound program coerces the same frozen weights into four cognitive
modes (solve, diagnose, advise, revise). Evolution should find prompts that
make each role maximally differentiated. Testable predictions:

- Evolved scaffolds should become more role-differentiated over iterations
  (measurable via role-classifier accuracy or embedding-cluster separation).
- Cross-role prompt swap should degrade performance (ablation: swap diagnose
  into advise slot).

### 3. Feedback engineering (the actual GEPA lever)

GEPA's claim is that **natural-language traces carry richer credit-assignment
signal than scalar reward**. Per-module `Feedback` strings — not accuracy —
are the substrate GEPA mutates against. Examples emitted by the adapter:

- `"OVER-ADVISING REGRESSION: draft was correct, advice emitted, revision
  broke the answer. Tighten suppression..."`
- `"MISSED REPAIR: draft wrong, advisor emitted NO_ADVICE. Add a diagnostic
  check for this failure type..."`
- `"SUCCESSFUL PRESERVATION: draft correct, NO_ADVICE, final correct.
  Preserve this suppression behavior."`

Ablation: compare GEPA with scalar-only, GEPA with generic per-example
feedback, GEPA with this advisor-specific typed feedback. If the typed
feedback isn't what's driving the lift, the contribution collapses.

## Experimental programme

| Run | Purpose | Expected finding |
|-----|---------|------------------|
| A | Actor only | baseline |
| B | GEPA actor-only (Asawa's static-GEPA setup) | static-prompt ceiling |
| C | Hand-written compound scaffold | shows cost of over-advising |
| D | GEPA full compound scaffold | the north-star |
| E | D without NO_ADVICE pathway | isolates trajectory-regularization effect |
| F | D with scalar-only feedback | isolates feedback-engineering effect |
| G | Trained Advisor (Asawa replication) | upper bound for compile gap |
| H | Reflect-then-distill: SFT small advisor on D's rollouts | the GEPA↔Advisor bridge |

Win conditions (ladder, in order of strength):
1. **Internal:** D > B on the same actor/split.
2. **Mechanistic:** D's NetReg > B's NetReg (trajectory-level signature,
   not just mean lift).
3. **Compilation:** $G_\text{compile} = J(G) - J(D)$ narrows compared to
   published Asawa numbers.
4. **Strongest:** D ≈ G (text ≈ weights as compilation target); or H ≥ G
   with less compute than GRPO.

## Domain choice: RuleArena Taxes, not MATH

MATH saturates on genre-familiarity. RuleArena tests **control of explicit
external rules, long-context attention, logical rule-disambiguation, and
numerical computation** — which is where advisor scaffolding has a job to do.
Using RuleArena Taxes complexity-0 (matches Asawa's reported setup).

## Status

Infrastructure: built (20/20 unit tests green). Compound program, adapter
with typed feedback, two arenas (Taxes, MATH ZPD), two variants (actor-only,
full scaffold), CLI over `gepa.optimize`.

**Preliminary evidence** (small-scale, same-model reflection, Qwen-7B actor):
on a signal-rich MATH subset (pass-rate ∈ [0.15, 0.50]), +13.3pp dev lift
on both variants. Full scaffold started 20pp below actor-only baseline
(confirming Asawa's over-advising) and its holdout gap was **–7pp vs
actor-only's –33pp** — preliminary evidence for the trajectory-regularization
claim.

**Running now:** Qwen2.5-72B-AWQ actor (capability-tier analog of GPT-4.1-
mini) on RuleArena Taxes, variants B and D, budget 80/160 metric calls.
Seed actor-only baseline = 1/15; seed full-scaffold baseline = 0/15. Over-
advising reproduces on a 72B actor.

## If it works

The contribution is conceptual, not benchmark: text-compiled advisor
policies approach weight-compiled ones in this regime. Positioning against
Sky Lab becomes "two compilation targets, characterized" not "we beat them."

## If it doesn't

Characterizing the compile gap is itself the result. Sky Lab has not
published this comparison; either outcome is new information.

## Compute envelope

One B200 spot instance. Qwen-72B-AWQ inference only, no training. Full
campaign (A through H, multiple seeds) fits in <24 hours wall and <$5.
