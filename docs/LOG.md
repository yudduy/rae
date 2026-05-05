# Genealogy of the work

A traceable log of the thinking, experiments, and findings that produced this codebase.

---

## 1. Thesis

A frozen LLM is not a fixed capability number. It is a conditional dynamical system whose trajectory distribution is selected by context. The scientific question is not "does prompting help" but:

> How much of a learned control policy for a frozen executor can be compiled into **text** (evolved prompts) rather than into **weights** (an RL-trained advisor)?

Two recent artefacts instantiate the two compilation targets for the same abstract object — a control policy over a frozen executor:

- **GEPA** (Agrawal et al. 2025, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)) — reflective prompt evolution. Compiles control policies into text. Beats GRPO with up to 35× fewer rollouts.
- **Advisor Models** (Asawa et al. 2025, [arXiv:2510.02453](https://arxiv.org/abs/2510.02453)) — a small advisor model trained with GRPO. Compiles control policies into weights. Closes a ~12pp gap on GPT-4.1-mini RuleArena Taxes (64.8% → 76.8%); reports +71% relative for GPT-5. Notably reports that *static GEPA on the actor prompt alone fails to recover the standalone baseline* on this task.

Define the compilation gap:

```
G_compile = J(π_weight) − J(π_text)
```

measured on the same actor, same data split, same inference harness.

- `G ≈ 0` ⟹ text is a sufficient representational substrate for advisor policies; training and evolution converge.
- `G > 0` ⟹ characterizes the **limits of context as a control medium** — i.e. the class of advisor behaviors that require amortization into parameters.

The empty cell, which neither paper occupies: GEPA applied to the *whole* Actor → Diagnose → Advise → Revise scaffold (not just the actor system prompt). That is what this codebase tests.

The deeper motivation comes from the **Dynamic Cheat Sheet (DCS)** lineage (Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou 2025, [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)): a frozen model maintains a persistent text memory of strategies it writes for itself, retrieves at inference, and reuses. DCS demonstrates the form works empirically (≈+27pp on AIME 2024 with Claude 3.5; Game-of-24 from ~10% to ~99% on GPT-4o). What DCS leaves open is *which property* of an entry causes the gain — fluency, similarity-match, structure, or actual strategy content. The advisor compilation question attacks the *when-to-add-an-entry* and *when-not-to-retrieve* sides of that curation problem.

---

## 2. Design — advisor-only GEPA on a frozen actor

The actor is frozen end-to-end. Weights stay at the published checkpoint; the actor's solve-time system prompt and revise-turn instruction are pinned to Asawa et al.'s `STUDENT_SYSTEM_PROMPT` and published revise-prompt chat layout. **GEPA evolves only the advisor**'s two prompts (`advisor_diagnose` and `advisor_advise`), using feedback from downstream task performance.

```
              ┌────────────────────────────────────┐
              │ actor_solve   (FROZEN)             │
problem ───▶ │   = Asawa STUDENT_SYSTEM_PROMPT    │ ─▶ draft
              └────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────────────────────────┐
draft + ──▶ │ advisor_diagnose   (GEPA-evolved)  │ ─▶ FAILURE_MODE + EVIDENCE  |  NO_DRIFT
problem      └────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────────────────────────┐
diagnosis ─▶ │ advisor_advise    (GEPA-evolved)   │ ─▶ concrete hint  |  NO_ADVICE
              └────────────────────────────────────┘
                       │
       ┌───────────────┴────────────────┐
       │                                │
   NO_ADVICE                         advice
       │                                │
       ▼                                ▼
   draft = final                ┌────────────────────────────────────┐
                                │ actor_revise   (FROZEN)            │
                                │   = Asawa revise instruction       │ ─▶ final
                                └────────────────────────────────────┘
```

`actor_revise` matches `advisor_models/rule_arena/env.py:_build_student_prompt` verbatim — `[system=actor_solve] [user=question] [assistant=draft] [user=advice + actor_revise]` — so any GEPA delta is attributable to the **advisor's text-compiled policy**, not to drift in the actor's instructions. This mirrors Asawa's setup as cleanly as possible: they hold the student fixed and train the advisor's *weights*; we hold the student fixed in every sense and evolve the advisor's *text*.

### Why two advisor modules, not one

Asawa's flow is three steps: actor solves → advisor advises → actor revises. The advisor in their setup is a single neural net producing advice from `(problem, draft)`. To give GEPA's reflection LM a cleaner target, we split the advisor's behaviour into two prompt-mutable components:

- `advisor_diagnose` — produces a structured `FAILURE_MODE + EVIDENCE` schema, or declares `NO_DRIFT`.
- `advisor_advise` — reads the diagnosis and either emits a concrete hint or `NO_ADVICE`.

*"The diagnose prompt should require concrete evidence before flagging a failure"* and *"the advise prompt should suppress when the diagnosis is weak"* are different mutations on different components. Splitting them lets GEPA evolve the *should-I-intervene?* decision and the *what-to-say?* decision independently — which is the surface the abstention reward actually needs.

### Per-module reflective feedback (the GEPA lever)

Per the GEPA paper §5 ("feedback engineering"), the *richness* of textual feedback fed to the reflection LM matters more than scalar reward. The adapter (`src/rae/gepa_adapter.py`) synthesizes per-module diagnostics:

- `actor_solve` — format-line missing, draft already correct (over-advising risk), wrong magnitude vs wrong sign.
- `advisor_diagnose` — missing FAILURE_MODE / EVIDENCE schema, generic critique rather than concrete error citation.
- `advisor_advise` — emitted advice on already-correct draft (over-advising regression), suppressed advice on wrong draft (failure to find leverage), advice too long (solution-leakage risk).
- `actor_revise` — regression (broke a correct draft), repair (fixed a wrong draft), or no-effect.

These are the strings the reflection LM mutates against.

### Trajectory regularization (the metric beyond accuracy)

Capability lives in a trajectory distribution, not a mean. Define the transition table per problem:

| Draft | Final | Name |
|-------|-------|------|
| wrong | right | repair |
| right | right | preservation |
| right | wrong | over-advising regression |
| wrong | wrong | failed repair |
| right | NO_ADVICE → right | successful abstention |
| wrong | NO_ADVICE → wrong | missed opportunity |

Headline metrics tracked alongside accuracy:

```
NetReg     = P(W → R) − P(R → W)
OverAdvise = P(R → W | advice emitted)
```

A good advisor scaffold is not maximally talkative. It is selectively causal — it must learn **silence** on correct drafts.

---

## 3. Empirical record

Two campaigns, both with strict GEPA-style train / dev / holdout discipline (GEPA evolves on train+dev; the *frozen* evolved candidate is then evaluated on a held-out set).

### 3.1 Signal-rich MATH (Qwen-7B actor) — preliminary

A pass-rate-bounded MATH subset (Qwen-7B pass-rate ∈ [0.15, 0.50]) where actor-only GEPA has signal.

**Actor-only GEPA**: dev 0.40 → 0.533 (+13.3 pp), holdout 0.20.

**Full 4-module compound GEPA**: dev 0.20 → 0.333 (+13.3 pp), holdout 0.267.

Holdout transition matrix for the full compound (n=15):

| Transition | Count | Rate |
|---|---|---|
| R → R (advice emitted) | 3 | 20% |
| R → R (NO_ADVICE) | 1 | 7% |
| R → W (over-advising regression) | 2 | 13% |
| W → R (repair) | 0 | 0% |
| W → W | 7 | 47% |
| W → W (NO_ADVICE) | 2 | 13% |

- Net regularization: −13.3 pp
- Repair rate: 0/15
- Over-advising rate: 2/12 = 16.7%
- NO_ADVICE precision: 1/3 = 33%
- NO_ADVICE recall: 1/6 = 17%

Both variants improved dev by the same +13.3 pp. The full scaffold's holdout drop was smaller (−7 pp vs −33 pp for actor-only), which is suggestive but not decisive at N=15.

### 3.2 RuleArena Taxes (Qwen-72B-AWQ actor) — strong actor

Direct test of the empty cell: GEPA on the full compound vs GEPA on actor-only, on the domain Asawa et al. report Advisor Models' biggest gains.

| Run | Variant | Budget | Seed dev | Best dev | Holdout | n_cands |
|---|---|---|---|---|---|---|
| Actor-only | GEPA evolves only `actor_solve` (replicates Asawa's static-GEPA baseline) | 80 | 0.067 | 0.067 | 0.067 | 5 |
| Full compound | GEPA evolves all four modules | 160 | **0.000** | 0.067 | **0.000** | 8 |

The full-compound seed score (0.000) is below the actor-only seed (0.067) — that is Asawa's over-advising regression reproducing on a 72B actor. GEPA recovered it to dev parity but did not exceed actor-only, and **failed to generalize to holdout**.

Holdout transition matrix for the best-dev full-compound candidate (n=15):

| Transition | Count | Rate |
|---|---|---|
| R → R | 0 | 0% |
| R → W (over-advising regression) | 1 | 6.7% |
| W → R (repair) | 0 | 0% |
| W → W | 14 | 93.3% |

- Net regularization: −6.7 pp
- Repair rate: 0/15
- Over-advising rate: 1/15 = 6.7%
- NO_ADVICE precision: undefined (0 silences)
- **NO_ADVICE recall: 0%**

The mechanically clear part: **the scaffold's recall on the should-stay-silent class was 0%.** The evolved compound advised on every single held-out problem, even when the actor's draft was already correct. It never learned when *not* to advise.

### 3.3 What the two campaigns showed

1. **The empty cell is real but GEPA-Advisor does not trivially win it.** Under same-model local reflection at modest budgets (≤160 metric calls), the GEPA-evolved full scaffold matches but does not exceed GEPA actor-only on RuleArena Taxes. Both plateau near the untrained-actor baseline. Asawa's reported failure of static-GEPA-on-actor reproduces; the 4-module scaffold doesn't fix it.

2. **The failure mode is mechanistically specific: GEPA does not learn silence.** Across both held-out sets, the evolved scaffold exhibits the same pathology — W→R near 0, NO_ADVICE recall low (0% on Taxes, 17% on MATH), over-advising 6–17% out of advice-emission events. Reflective evolution finds scaffolds that *try to help every problem*, not scaffolds that *detect when to stay silent*. Text-compiled intervention is cheap; text-compiled abstention is harder, because the reward signal doesn't directly reinforce it (suppression only shows up as preservation, R→R via NO_ADVICE).

3. **Holdout generalization is poor at N=15.** Strong dev → near-zero holdout. Statistical-power issue compounded by dev-specific overfitting.

4. **Same-model local reflection may be a hard ceiling.** Actor-only GEPA produced 5 candidates in the Taxes run; none beat the seed. Consistent with Qwen-72B reflecting on Qwen-72B's own traces — the reflection model can't see failure modes the task model itself doesn't already know about.

### 3.4 Implication for the compile gap

We can't directly measure `G_compile` yet — we don't have an Asawa-trained-advisor number on our actor. But the transition matrix exposes a substructure of the gap:

The GEPA-evolved text scaffold *does not encode the silence/abstention behavior* that a weight-trained advisor presumably learns. Asawa's trained advisor reaches +12 pp over baseline on this domain; our GEPA scaffold reaches 0 pp, with a scaffold that advises on 100% of holdout. If that 12 pp is mostly the learned-silence component, then **text compilation is sufficient for advice generation but insufficient for advice gating** — a cleaner and more specific claim than "GEPA < Advisor Models."

---

## 4. Activation-space pivot — the advisability probe

Mid-quarter pivot: instead of evolving text-space scaffolds, can we read intervention value directly from the actor's hidden states? The hypothesis:

> **advisability** — the expected value of intervening on a draft — is linearly readable from the frozen Actor's residual stream.

If so, a probe trained on hidden states could gate advice (emit when probe says "this will help," abstain otherwise) and resolve the abstention failure observed in §3.

### 4.1 Forced-advice paired rollout pilot (Qwen-7B + MATH ZPD, n=250)

Setup: forced-advice paired rollouts (every problem gets advice, no NO_ADVICE pathway), to generate transition labels for probe training.

| Quantity | Value |
|---|---|
| draft_acc | 0.280 |
| final_acc (forced advice) | 0.272 |
| net regularization | **−0.008** (advice ≈ neutral, slightly damaging) |
| W→W transitions | 167 |
| W→R (repair) | 13 |
| R→R (preserve despite forced advice) | 55 |
| R→W (over-advising regression) | 15 |
| Oracle ceiling accuracy | 0.332 (+5.2 pp over never-advise) |

Forced advice was approximately neutral on MATH ZPD. The oracle ceiling (+5.2 pp) is the upper bound an ideal gating policy could recover; the actual forced-everywhere policy doesn't capture it.

### 4.2 Probe results

Linear probes on the actor's hidden states at every layer × position × method (LR+L2, etc.):

| Probe | Best AUROC | 95% CI |
|---|---|---|
| Probe-R: P(R → W \| draft_correct, h) at layer 1 | **0.745** | [0.601, 0.852] |
| Probe-W: P(W → R \| draft_wrong, h) at layer 5 | 0.664 | [0.527, 0.797] |
| Surface-feature baseline (correctness logprob) | 0.665 | [0.591, 0.729] |

Initial readout: probe meaningfully beats surface baseline at the over-advising-detection task (AUROC 0.745 vs 0.665).

### 4.3 Three findings that killed the thesis

**(a) Probe doesn't beat logprob at gate-level decision-making.** The downstream metric — gating advice using the probe vs gating using the surface logprob threshold — gave Δ = +0.008 (within 0.27σ bootstrap noise). The "probe > surface" falsifier committed to before running didn't pass.

**(b) The AUROC = 0.745 is a question-feature confound, not actor-state signal.**
- `prompt_last` (final chat-template token *before* draft generation) at layer 1 → AUROC **0.743**.
- `gen_last` (final token *after* the full draft is generated) at layer 1 → AUROC **0.745**.

These are indistinguishable. If advisability were a property of the actor's *dynamic state*, `gen_last` (which contains everything `prompt_last` does plus the full draft trajectory) should dominate. It doesn't. The probe is reading question-difficulty features accessible *before the draft exists*, not actor reasoning.

**(c) Multiple-testing inflation.** The search was 29 layers × 2 positions × 2 methods ≈ 116 configurations, with the maximum AUROC reported. Under the null hypothesis (no real advisability signal), the family-wise probability of seeing some configuration hit ≥ 0.745 is approximately 16–20%. The naive 95% CI [0.601, 0.852] is misleading because bootstrap doesn't correct for selection across the search.

### 4.4 Honest verdict

The activation-probe advisability thesis did not survive the pilot. The intended decisive follow-up — a K-draft within-problem cross-actor experiment that would distinguish question-feature signal from actor-state signal — was launched but did not complete due to infrastructure failure during overnight runs.

What's defensible from this pilot:

1. **Formalization of advisability** = E[Δ_advise | h] = J(with advice) − J(without advice | h) as a target distinct from draft correctness.
2. **Counterfactual transition matrix** on forced-advice paired rollouts as a measurement template.
3. **The probe-gated vs surface-gated comparison** showing the gap is small at this scale.
4. **The question-feature confound** identified via the `prompt_last ≈ gen_last` coincidence — a methodological caution future activation-based gate work should pre-register against.

---

## 5. Methodological audit (added 2026-05-04)

The empirical regime in §3 and the probe campaign in §4 both held the experiments to **GEPA-style train / dev / holdout discipline**: GEPA (or the probe) optimizes on train+dev, then the *frozen* evolved candidate (or trained probe) is evaluated on a held-out set the model has never seen.

That regime is **stricter than the DCS lineage we set out to extend**. DCS is fundamentally an *online learning* claim — the cheatsheet accumulates *during* evaluation, so frozen-then-test discipline does not test the DCS hypothesis at all. The proper DCS-style evaluation — let the scaffold continue evolving across the eval stream, with each new problem updating the evolved prompts via the same reflection loop — has not been run on this codebase.

The strict-holdout failure here should be read as **"frozen evolved scaffold doesn't transfer under strict discipline"**, not as **"text-space steering is broken in the DCS regime."**

---

## 6. What's open

### Re-run under DCS-style online learning
Let the 4-module scaffold continue evolving across the held-out problems, sequentially, with each new example updating the evolved prompts via the same reflection loop. This is the experiment that would actually test the DCS-extension hypothesis on top of the compound-program substrate.

### Missing intermediate ablations
The runs to date have been actor-only GEPA vs full 4-module GEPA. Three missing intermediates would isolate which mechanism is load-bearing:

- **Hand-written compound (no GEPA).** Establishes the cost of over-advising before evolution acts.
- **Asawa replication on the same actor.** Establishes the upper bound — what a *weight-compiled* advisor achieves on Qwen-72B-AWQ — so `G_compile` can be computed directly.
- **2- or 3-module GEPA.** Isolates whether the advisor → diagnose+advise split is load-bearing or whether a simpler compound suffices.

### NO_ADVICE-loss training signal
Modify the per-module Feedback strings to weight suppression pathways more strongly: `R → R via NO_ADVICE` is "ideal preservation," `W → W via NO_ADVICE` is "missed opportunity," `R → W` is "regression." Current feedback engineering may not weight these strongly enough relative to the simpler "advise more concretely" mutation direction.

### Stronger reflection LM
Same-model reflection (Qwen-72B reading its own traces) may be a hard ceiling — the reflection model can't see failure modes the task model itself doesn't already know about. The GEPA paper's gains rely on reflection-quality. An external reflection LM (Claude / GPT-4-class) would test whether the cap is reflection capacity vs scaffold representational capacity.

### Larger holdout
N=15 is too small for a defensible holdout claim. 50–100 problems before any generalization claim holds.

### Activation-probe follow-ups
The K-draft cross-actor experiment that would have distinguished question-feature signal from actor-state signal still hasn't run. If pursued, it needs a stronger asymmetry between actor and advisor (e.g. Qwen-72B actor + Claude / GPT-4 advisor) to break the self-correction confound flagged during the pilot review.

---

## 7. Cited literature

The work in this repo builds directly on:

- Asawa, P., Zhu, A., Zaharia, M., Dimakis, A. G., & Gonzalez, J. E. (2025). *How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models.* arXiv:[2510.02453](https://arxiv.org/abs/2510.02453). Code: [github.com/az1326/advisor-models](https://github.com/az1326/advisor-models).
- Agrawal, L. A. et al. (2025). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* arXiv:[2507.19457](https://arxiv.org/abs/2507.19457). Code: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa).
- Suzgun, M., Yuksekgonul, M., Bianchi, F., Jurafsky, D., & Zou, J. (2025). *Dynamic Cheat Sheet: Test-Time Learning with Adaptive Memory.* arXiv:[2504.07952](https://arxiv.org/abs/2504.07952).

Adjacent / contextual:

- Yuksekgonul, M., Koceja, D., Li, X., Bianchi, F. et al. (2026). *Learning to Discover at Test Time (TTT-Discover).* arXiv:[2601.16175](https://arxiv.org/abs/2601.16175).
- Liang, W., Sun, Y., Nan, S., Li, C., Song, D., & Kawaguchi, K. (2026). *Strategy Executability in Mathematical Reasoning.* arXiv:[2602.22583](https://arxiv.org/abs/2602.22583).
- Wang et al. (2024). *RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios.* arXiv:[2412.08972](https://arxiv.org/abs/2412.08972).
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). *Are Emergent Abilities of Large Language Models a Mirage?* arXiv:[2304.15004](https://arxiv.org/abs/2304.15004).
