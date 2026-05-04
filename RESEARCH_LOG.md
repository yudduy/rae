# Research log — Advisability Probing for LLM Advisors

Operating in autonomous research-loop mode. Pattern per iteration:

1. **Build the next falsifiable step.** Minimal code, TDD where cheap, deploy to B200.
2. **Run on B200.** vLLM Qwen-72B-AWQ at localhost:8001; HF cache at /home/ubuntu/capr-cache/hf/; Qwen-7B available.
3. **Inspect the result.** If it looks right: proceed. If it looks wrong or negative: stop and diagnose.
4. **On negative / debug / anomaly:** launch a literature subagent (Explore/general-purpose with alphaxiv + DeepWiki + WebSearch) to see if the pathology is named, fixed, or preempted. Synthesize under 800 words.
5. **At decision forks:** consult ChatGPT Pro via `chatgpt-pro-web "…" --model gpt-5-pro -o /tmp/cgpt.md --timeout 30`. Instruct it to be skeptical and verify the math.
6. **Update this log** per iteration: what ran, what it showed, what ChatGPT Pro/agents said, what changed.
7. **Continue** until conference-paper-grade result.

Core hypothesis under test: **advisability** — the expected value of intervening on a draft — is linearly readable from the frozen Actor's residual stream. Over-advising is a perception failure, not a behavior failure.

---

## Iteration index

Each iteration gets a numbered entry: date, goal, action, result, decision.

### Iter 00 — Setup + harness audit (2026-04-20)

**Goal.** Confirm RuleArena Taxes harness matches Asawa. Confirm B200 state. Commit to advisability framing.

**Action.** Diffed `src/rae/arenas/rule_arena_taxes.py`, `src/rae/evaluator.py`, `src/rae/seed_prompts.py` against `references/advisor-models/baselines/gepa/gepa_rule_arena.py` + `advisor_models/rule_arena/config.py`. Inspected `data/rule_arena/train_gpt-4.1-mini_0.parquet` + `validation_gpt-4.1-mini_0.parquet`. Queried capr-diag via `flow ssh`.

**Result.**
- Scorer numerically equivalent (same regex, `np.isclose`, overpaid→negative).
- Same 100 raw problems.
- Seed actor prompt = `STUDENT_SYSTEM_PROMPT` verbatim.
- Asawa parquet shows gpt-4.1-mini `initial_reward` = 57/100 = 57% (paper's 64.8% is a later eval).
- Qwen-72B-AWQ at 1/15 ≈ 7% is real, not a harness bug. Quantized 72B on Taxes ≪ frontier.
- B200 up, 15.5GB VRAM free alongside vLLM.

**Decision.** 1/15 baseline is real. Harness is sound. Start probe experiment on **Qwen-7B + MATH ZPD** (we have pass_rate-filtered data with both correct and incorrect drafts, needed for probe training). Defer Qwen-72B-AWQ activation extraction.

### Iter 01 — Paired-rollout pipeline + lit verify (2026-04-20)

**Goal.** Build paired-rollout + activation-capture script. Verify approach against current probe literature before deploy.

**Action.** Wrote `src/rae/activation_actor.py` (HF transformers wrapper with `chat_and_capture`) and `src/rae/collect_advisability.py` (per-problem driver writing jsonl + npz). Launched lit-verify subagent (alphaxiv + DeepWiki + WebSearch).

**Result — activation capture mechanics.** `output_hidden_states=True` via a second clean forward pass on `[question||draft]` is the canonical approach. `hidden_states` is length `n_layers+1`; index 0 is embedding, index i≥1 is post-block-i residual; index -1 is post-final-RMSNorm on Qwen2. Gotchas: (a) must use second forward pass, NOT `generate(..., output_hidden_states=True)` — matches our code. (b) padding/mask must be correct. Cencerrado et al. (arXiv:2509.10625) use exactly this protocol on Qwen2.5-7B-Instruct — we're on-protocol.

**Result — novelty.** Verdict: weakly novel, adjacent work is dense. Three near-neighbors to differentiate against:
- Zhang et al., "Reasoning Models Know When They're Right" (arXiv:2504.05419) — correctness probe gating early-exit. Closest prior art.
- Cencerrado et al. (arXiv:2509.10625) — Qwen2.5-7B-Instruct pre-generation correctness probe explicitly for abstention/fallback.
- Lugoloobi et al. (arXiv:2602.09924, year-2026 ID, treat as unverified until checked) — probe for model-routing/cascade escalation.
Three real differentiators: (1) label = causal treatment effect J(with advisor) − J(without advisor), NOT correctness. (2) property of the executor–advisor pair. (3) predicts SIGN of Δ (R→W regression detection is beyond any correctness probe). Frame the paper on these or reviewers shrug.

**Result — statistical power.** 200 problems is a **pilot**; 500 is publishable-grade. MATH ZPD 4-way label (WW, RR, WR, RW) has minority classes (R→W ≤ 5%, ≈10 samples at n=200) that are severely underpowered for a 3584-dim probe. Mitigations from Cencerrado + David + Sun: (a) collapse to BINARY target (sign of Δ, i.e., "advice helped vs. not") as primary probe; (b) per-layer linear probe + L2 regression; (c) difference-of-means probe (Cencerrado zero-free-parameter baseline — works at n=160 for correctness); (d) bootstrap CIs over problems; (e) NO MLP, NO blind PCA at n=200. If DoM-AUROC > 0.65 on held-out fold at n=200, scale to 500.

**Decision.** Proceed with the 200-problem MATH ZPD pilot. Adjust probe strategy per above (binary-Δ primary + DoM baseline + per-layer ridge). Document expected outcomes: pilot-threshold = 0.65 AUROC on binary sign(Δ) with DoM probe at best layer. Also flag: Lugoloobi 2602.09924 needs verification before cite.

**Open risks.** (a) At temperature 0 on Qwen-7B, the scaffold may emit advice on every problem (no NO_ADVICE diversity), collapsing the label distribution. If so, need to either sample at temp>0 or change the advise prompt to enforce abstention on high-confidence drafts. (b) GEPA-style seed candidates may already encode a weak abstention policy we'll confound with probe signal — baseline must include "always advise" branch for fair comparison.

### Iter 02 — Council deliberation on probe design (2026-04-20)

**Goal.** Stress-test the experimental design against Codex + Gemini + Claude before committing overnight compute.

**Action.** Ran `deliberate` with the full pilot spec + RESEARCH_LOG + collect_advisability.py as context. Claude subagent unavailable; got Codex and Gemini verdicts.

**Key convergent finding (CRITICAL).** Both models identified the same fatal flaw in the primary target. `y = sign(Δ > 0) = 1[draft_wrong AND final_right]` is **mathematically equal** to `1[W→R transition]`. A probe on this pooled label can achieve high AUROC by simply reading draft-correctness — no advisability information required. Codex: "probe can look good by reading draft correctness, not intervention value." Gemini: "it will likely correlate more with Input Correctness than with advice efficacy; call it a Repairability probe rather than Advisability probe."

**Required fix.** Condition on draft correctness and train two separate probes:
- **Probe-W**: `P(W→R | draft_wrong, h)` — which wrong drafts are rescuable?
- **Probe-R**: `P(R→W | draft_correct, h)` — which correct drafts will be damaged?

The advisability claim is only supported if BOTH probes beat baselines (draft-token logprob, random, always-advise). A correctness probe reaches Probe-W but cannot reach Probe-R — that's the novelty litmus.

**Gemini's other warning ("Self-Correction Confound").** Actor advising Actor — the probe may learn `high_entropy → advise` as a proxy, not a causal understanding of advisability. Mitigation: in follow-up, use a different advisor model (e.g., Qwen-72B advising Qwen-7B) to break the symmetry.

**Convergent specs.**
- Data collection unchanged (paired rollouts, both `prompt_last` and `gen_last`, temperature 0). ✓ matches our pipeline.
- Probe training changes: conditional targets, per-layer linear probe + L2 ridge, DoM baseline (Cencerrado recipe), bootstrap CI over problems.
- Scale bar: AUROC ≥ 0.72 on held-out fold with problem-bootstrap 95% lower bound > 0.55. Take Gemini's tighter number. Anything 0.67–0.72 = inconclusive; < 0.67 = kill or pivot.

**Decision.** Deploy collection pipeline tonight as-is (paired-rollout, 200 MATH ZPD). Write probe training to use conditional targets, not pooled sign(Δ). Include prompt-last + gen-last both. Use DoM baseline as the paper's cleanest claim.

**What this changes about the paper.** Two-probe story is sharper than pooled-sign story. Probe-R (R→W prediction) is the one no prior correctness probe achieves — it's the wedge against Zhang et al. 2504.05419 and Cencerrado 2509.10625. Lead with Probe-R as the main result.

### Iter 03 — Smoke tests and design pivot (2026-04-20)

**Goal.** Get the pipeline end-to-end on B200; diagnose any design issues before the overnight run.

**Action.** Smoke-tested on 3 MATH problems inside the vLLM container; hit CUDA OOM because vLLM 72B was holding 163GB of 183GB. Created a parallel `rae-pilot` container on the same B200; stopped the `capr-actor` vLLM container to free all 183GB. Re-ran 3-problem smoke test.

**Result — smoke_v2 (default seed, 3 problems).** 3/3 drafts wrong. 3/3 advisors output `NO_ADVICE`, even though the advisor's DIAGNOSIS step correctly identified real flaws in each draft. All 3 transitions = W→W via preserve branch. Advisor never speaks → zero Δ label diversity → probe training cannot distinguish useful advice from noise.

**Diagnosis.** At T=0 with same-model reflection, the `ADVISOR_ADVISE_SEED_MATH` prompt's exit clause ("If the diagnosis is weak or the draft already looks correct, output exactly: NO_ADVICE") dominates the advisor's output. Qwen-7B defaults to the safe exit path. This is the mirror-image pathology of Exp5 on Taxes (100% advice, 0% NO_ADVICE) — a prompt-specific abstention policy that masks the true intervention-value signal.

**Decision (design pivot).** Separate intervention-value measurement from abstention policy into two training objectives:
1. **Collection with forced advice** (this iteration): force the advisor to always emit a concrete hint (never NO_ADVICE). Always run the revise step. This gives a clean paired rollout `Δ = final_revise_correct - draft_correct` per problem. The probe is trained on this uncensored Δ.
2. **Abstention as a downstream gate** (future iteration): given the trained advisability probe + the forced-advice data, learn a threshold for NO_ADVICE emission. This is a separate, lighter training problem.

Added `ADVISOR_ADVISE_SEED_MATH_FORCED` + `forced_advise_seed_candidate_math()` in `seed_prompts_math.py`; added `--force-advise` and `--force-revise` CLI flags in `collect_advisability.py`. Kept all existing tests green (6/6 pass).

**Result — smoke_v3 (forced advise+revise, 10 problems).** draft_acc=0.10, final_acc=0.30, **net_reg=+0.20**. Transitions: 7 W→W, 2 W→R, 1 R→R, 0 R→W. Wall time ~18s/problem.

This is unambiguously positive signal: on 10 MATH ZPD problems, forced Qwen-7B advice **rescues 2/9 wrong drafts** (W→R = 22%) without damaging the single correct draft. Over-advising doesn't dominate this regime; repairability does. Implication for the probe: Probe-W (P(W→R | draft_wrong, h)) carries the paper's main result; Probe-R is testable only if the full 250 has enough R→W cases.

**Decision.** Launch full 250-problem run with `--force-advise --force-revise --max-new-tokens 1536`. Estimated wall time 75 min (background process PID 306 in `rae-pilot` container). Build probe training code in parallel.

**Budget check.** B200 at ~$0.01/hr off-peak. Pilot run: ~1.5 h = $0.015. Full pipeline including probe training: under $0.10 total. No budget constraint.

### Iter 04 — While pilot runs (2026-04-20)

**Goal.** Build the probe training + gate evaluation code. Prioritise the "probe vs surface" falsifier per Codex consult.

**Action.**
- Wrote `src/rae/probes.py` — conditional probes (Probe-W = P(W→R | wrong), Probe-R = P(R→W | correct)) + correctness baseline. DoM (Cencerrado recipe) + L2 logistic regression. 5-fold CV + problem-level bootstrap CI. Per-layer + per-position (prompt_last, gen_last).
- Wrote `src/rae/gated_eval.py` — counterfactual gate simulation. Gates: always, never, oracle, random-oracle-rate, logprob-mean/last/min @ matched rate, probe-cascade @ best layer per position.
- Wrote `src/rae/surface_features.py` — post-hoc teacher-forced draft logprob extraction (mean, last, min, perplexity, len). The key falsifier: probe must beat this.
- Test coverage now 37/37 pass (added `test_probes.py`, `test_gated_eval.py`, `test_surface_features.py`).

**Consulted Codex** (`codex exec`) for ROI ranking of next-build options. Ranked: (C) surface baselines > (A) temporal analysis > (D) lit positioning > (B) 72B extraction > (E) cross-actor transfer. Reason: advisability only survives if probe beats logprob. Built (C).

**Pilot progress snapshot** (450s elapsed, 20/250):
- draft_acc = 0.10, final_acc = 0.25, net_reg = +0.15, advice_rate = 1.0 (forced)
- Wall time: 22.5s per problem. ETA ~94 min total.
- Trend: net_reg dropping from +0.40 → +0.20 → +0.13 → +0.15 (stabilising as sample grows).

**Decision.** Let the collection finish, then run surface_features + probes + gated_eval in that order. If probe AUROC on Probe-W and Probe-R beats logprob gates at matched advice rate, scale to 500 problems and write the paper. If not, the finding is: advisability is not more linearly readable from Qwen-7B residuals than draft confidence is — pivot to a different actor scale or different target.

### Iter 05 — Midpoint check (2026-04-20, 34 min into run)

**Goal.** Check pilot progress and update expectations.

**Observation.** 105/250 problems done. Stable aggregate: draft_acc=0.257, final_acc=0.229, **net_reg=-0.029**. The early-sample +0.40 net_reg at n=5 was small-n noise; at n=105 forced advice is slightly harmful on net.

**What this means.**
1. **Expected** given Huang 2023 (arXiv:2310.01798) and Xu 2024 (arXiv:2402.11436): self-correction without strong external feedback tends to hurt on reasoning tasks. Qwen-7B advising Qwen-7B is exactly this regime.
2. **Makes the probe thesis sharper, not weaker.** The gate question becomes: can the probe learn WHEN advice helps, even though on average it hurts? A probe-gated advisor that matches oracle within some gap would be a real result. A probe that tracks logprob would be a null.
3. **Transitions breakdown (predicted from net_reg=-0.029 at n=105):** W→R ≈ R→W − 3 ≈ small minorities (say 10–15 each); W→W dominant; R→R moderate. Still enough positives for both conditional probes if the final n=250 follows this distribution.

**Estimated at completion (n=250):**
- W→R count: ~20–30 (Probe-W training set)
- R→W count: ~25–35 (Probe-R training set)
- Minority class sizes borderline for stable linear probes at 3584-dim; DoM baseline will be more reliable than LR+L2.

**Pacing.** 22.5s/problem × 250 = ~94 min total. ~48 min remaining. Next wake-up at ~175 problems.

### Iter 06 — Pilot complete + analysis pipeline (2026-04-20, ~90 min wall)

**Pilot final result** (n=250):

```
draft_acc          = 0.280   (70/250 correct)
final_acc          = 0.272   (68/250 correct)
net_reg            = -0.008  (advice ≈ neutral, very slightly harmful)
advice_rate        = 1.0     (forced)
transitions:
  W->W = 167   (wrong draft, advice didn't help)
  W->R =  13   (repair: advice rescued a wrong draft)
  R->R =  55   (correct draft, survived forced advice)
  R->W =  15   (over-advising regression: advice broke a correct draft)
```

**Observations.**

1. **Forced advice on MATH ZPD + Qwen-7B is ~neutral.** 13 repairs versus 15 regressions — advisor is as likely to damage as to help. Consistent with Huang 2023. This is the realistic regime for a probe to solve.
2. **Probe training set sizes (tight but workable).**
   - Probe-W (P(W→R | draft_wrong, h)): n=180 with 13 positives (7.2%). Minority class severely underpowered at 3584-dim; DoM more reliable than LR+L2.
   - Probe-R (P(R→W | draft_correct, h)): n=70 with 15 positives (21.4%). Minority class tiny (15 samples) — results near the noise floor unless signal is very strong.
   - Correctness baseline: n=250 with 70 positives (28%). Balanced enough.
3. **Oracle ceiling** (advise iff Δ>0, via paired data): rescue 13 W→R without damage, skip 15 R→W and all 50 R→R_preserve. Accuracy = (draft_acc × (no_damage_count) + final_acc_where_helps) / 250. Quick math: 70 correct - 15 R→W saved by skipping on R + 13 W→R from advising on W = 68 correct at always-advise, 70 at never-advise, 68+15 = 83 at oracle = **accuracy 0.332**. So oracle gap over baselines is **+5.2pp over never-advise, +6pp over always-advise**.

**Analysis pipeline triggered.**
- ✓ `surface_features.py` — done, 250/250 ok.
- ⏳ `probes.py` — running (long wall time because of 29 layers × 2 positions × 3 targets × 2 methods × 5-fold LR).
- ⏳ `gated_eval.py` — queued.

**Thresholds I committed to up front (for self-audit).**
- Probe-R AUROC ≥ 0.72 → strong signal, scale to 500.
- Probe-R AUROC 0.67–0.72 → inconclusive; add paired K=4 sampling for distributional Δ.
- Probe-R AUROC < 0.67 → null, pivot.
- Gate decision: probe-cascade accuracy must beat best logprob gate at matched rate.

### Iter 07 — Probe results (2026-04-20, n=250)

**Results (per-layer 5-fold CV + problem-level bootstrap; bolded = best of family):**

| Probe | Method | Position | Layer | AUROC | 95% CI lo | 95% CI hi |
|-------|--------|----------|-------|-------|-----------|-----------|
| **Probe-R (R→W)** | **LR+L2** | **gen_last** | **1** | **0.745** | **0.601** | 0.852 |
| Probe-R (R→W) | LR+L2 | prompt_last | 1 | 0.743 | 0.609 | 0.838 |
| Probe-R (R→W) | DoM | gen_last | 1 | 0.678 | 0.515 | 0.819 |
| Probe-W (W→R) | LR+L2 | prompt_last | 5 | 0.664 | 0.527 | 0.797 |
| Probe-W (W→R) | DoM | prompt_last | 13 | 0.595 | 0.423 | 0.763 |
| Correctness | LR+L2 | prompt_last | 6 | 0.665 | 0.591 | 0.729 |
| Correctness | DoM | prompt_last | 20 | 0.630 | 0.547 | 0.713 |

**Headline.**
1. **Probe-R lands at AUROC 0.745**, bootstrap lower bound 0.601 (above chance). This clears the pre-committed 0.72 threshold for "strong signal."
2. **Probe-R (0.745) ≫ correctness probe (0.665).** The hidden state encodes **intervention-damage risk** more strongly than draft correctness itself. This is the novelty wedge: no prior correctness probe explains this.
3. **Probe-W lands at 0.664** — weaker than Probe-R. Advice-repair is harder to predict than advice-damage, at least on this data.
4. **Best layer is layer 1** (both positions). Deserves scrutiny — layer 1 is near the embedding, so the signal may be in problem tokens rather than model computation.

**Red flag to audit next: layer-1 confound.**
If the damage-risk signal is linearly readable from layer 1, it might be encoded in the **question structure** (a problem property) rather than the **actor state**. Tests to run:
- Question-only probe: train a probe on hidden states of just the question tokens (no draft). If AUROC ≥ 0.7, the signal is in the problem, not the actor.
- Draft-length / surface-feature controls in gated_eval (in progress).
- Per-layer robustness: does Probe-R degrade smoothly away from layer 1 or is it brittle?

**Context window check (every 5th iteration).** We're at Iter 07. Re-reading the plan's Principles + Boundaries: (a) no benchmark-chasing, the goal is a clean mechanistic claim; (b) the paper survives if the claim is "advisability is linearly readable" OR "it isn't but here's why" — both are publishable. No drift.

**Gated_eval launched** (background). Waiting for gates to finish before deciding next experiment.

### Iter 08 — Layer-1 confound, in-data (2026-04-20)

**Immediate re-interpretation, no new code needed.**

The layer-1 result already contains its own control. `prompt_last` is the final chat-template token **before** draft generation — essentially a question-only representation. `gen_last` is the final token of the full draft. At layer 1:

- Probe-R LR+L2 @ `prompt_last` layer 1: AUROC **0.743** [0.609, 0.838]
- Probe-R LR+L2 @ `gen_last`    layer 1: AUROC **0.745** [0.601, 0.852]

These are indistinguishable. **If advisability were a property of the actor's dynamic state, `gen_last` (after the actor processed the full draft) should dominate `prompt_last` (before the draft even exists).** It doesn't. All the discriminative information is already present *before the draft is generated*.

**Honest interpretation.** Probe-R at layer 1 is reading **problem difficulty for forced-advice refinement**, not actor dynamic state. It's a *preemptive* classifier over the question, not a readout of the actor's advisability.

This collapses part of the thesis:
- ❌ "The actor's hidden state encodes whether its draft is advisable" — not supported by this data.
- ✓ "Forced-advice damage risk is predictable from the problem + 1 layer of model processing" — supported at AUROC 0.74, but weaker and less novel. Close to standard "problem hardness" prediction.

**What survives.** The weaker claim still has utility: a compute-cheap preemptive gate on problem text can decide whether to invoke forced-revision without generating a draft first. Saves inference cost. But it's no longer "advisability probing" in the intervention-value sense — it's closer to "learned-hardness classifier."

**What doesn't survive as stated.** The paper pitch "over-advising is gateway-blindness, fixed by activation probing on the frozen executor" — not supported by gen_last ≈ prompt_last.

**What to do.** Don't abandon yet. Two remaining questions:
1. Does the probe-cascade *gate* actually beat logprob? That's the utility test. If yes, the preemptive-gate paper is worth writing.
2. Is the signal LARGER at later layers with more data? 15 positives is too few. A 500-problem extension with the same forced-advise protocol might reveal middle-layer advisability beyond the layer-1 question feature. Worth running while we wait.

**Deferred.** Task #37 "question-only probe" is now redundant — the existing `prompt_last` IS the question-only control. Closing it.

### Iter 09 — Bootstrap-CI validity is broken at our (n, p) (2026-04-20)

**WebSearch finding:** bootstrap CI validity for logistic-regression AUROC requires **p = o(n)**. At my numbers (p=3584, n=70), p ≫ n. The reported CI [0.601, 0.852] is **almost certainly undercovering**. Reviewer with a stats background will attack this immediately.

Evidence:
- "Validity boundary is around p = o(n) for logistic regression, whether or not regularization is added" ([Bootstrap in High Dim, arXiv:2210.10974](https://arxiv.org/html/2210.10974)).
- "No fast reliable method for computing confidence intervals currently exists, especially for small datasets" ([Harrell, bootcal 2025](https://www.fharrell.com/post/bootcal/)).

**Re-reading my result honestly:**
- Probe-R DoM (zero free params) @ gen_last layer 1: AUROC **0.678** [0.515, 0.819]. Lower bound 0.515 barely > chance.
- Probe-R LR+L2 @ gen_last layer 1: AUROC **0.745** [0.601, 0.852]. This is what the bootstrap-undercoverage issue attacks most directly.

**The DoM result is what survives reviewer scrutiny.** DoM is parameter-free (one mean difference per dim) so high-dim isn't the same pathology. But AUROC 0.678 with lower-bound 0.515 is much weaker than 0.745.

**Two required additions before claiming significance:**
1. **Label-shuffle permutation test.** Randomize R→W labels 500 times, retrain LR+L2 at the same layer, record AUROC. If observed 0.745 exceeds 95% of permuted AUROCs → real signal. This is the canonical non-parametric alternative to bootstrap when p ≫ n.
2. **Scale to n≥500.** Brings p/n ratio closer to the validity boundary. Also tightens minority-class counts (15 R→W at n=250 → ~30 at n=500).

**Decision.** While gated_eval and ChatGPT Pro are both pending, build:
(a) permutation-test extension to `probes.py`, and (b) launch a second 250-problem collection run to double n to 500 tonight. Compute is free; time isn't the binding constraint (monitor will ping when things finish).

### Iter 10 — ChatGPT Pro verdict on layer-1 + next experiment design (2026-04-20)

**Pro says: B/C first, A distant third.** The Probe-R AUROC 0.745 is **not clean evidence for actor-state advisability**.

Key analytic points:
1. **Multiple-testing obliteration.** We searched 29 layers × 2 positions × 2 methods ≈ 116 configs and reported the max AUROC. Under null, family-wise probability of seeing ≥ 0.745 is **~16-20%** (even though naive single-config p-value is ~0.0015). Our reported result is *not* significant after max-test correction.
2. **Effective degrees of freedom for L2-LR** is bounded by `n_train`, not `p=3584`. So the parametric dim-vs-sample argument is less damning than I thought — but *selection bias* from searching over 116 configs is what actually kills it.
3. **Layer 1 being best** is not a Qwen2.5 architecture violation (first attention can aggregate a lot of context) but "for advice will flip a correct draft to wrong, I would expect stronger dependence on draft/reasoning/uncertainty states in middle/late layers." Red flag for B or C, especially since correctness baseline is already 0.665.

**Proper tests ChatGPT Pro prescribes (in order):**
- **Full-pipeline permutation max-test.** Permute R→W labels 500 times, rerun ENTIRE pipeline (all layers, all positions, all methods, all CV folds), record MAX AUROC per permutation. Compare observed max (0.745) to max-distribution. My current `--permutation` flag tests only per-config; insufficient. Need to rewrite.
- **Prospective test.** Freeze (layer=1, position=gen_last, method=lr_l2, C=1.0) from THIS study. Test once on a new held-out problem set (v2 is collecting). No re-selection allowed.
- **Within-problem variation (the decisive A-vs-B experiment).** Generate K drafts per problem at temperature > 0. A question-only detector classifies all K drafts of the same problem identically. An actor-state probe distinguishes drafts that become R→W from drafts that survive. **This cannot be confounded with question difficulty.** It's the clean test.

**Updated plan:**
- ✓ **Naive per-config permutation test** (currently running on v1) — minimum first check. Will give per-config p-values but NOT multiple-testing correction.
- ✓ **v2 collection n=500** (running) — tightens CI, also serves as a prospective test for Probe-R at the frozen config.
- 🔜 **Full-pipeline max-permutation test** — rewrite `--permutation` to shuffle labels and re-select across layers/positions/methods per shuffle. Expensive (~45 min for 200 perms × LR-L2-only) but definitive.
- 🔜 **K-draft within-problem experiment** — new collection: temperature 0.6, K=4 drafts per problem, run forced-advise on each, label per-draft transition. Clean A-vs-B test. Needed before any paper claim about "actor state" vs "question property".

**Budget check.** Still well under $10. v2 ~$0.03. K-draft run would be 4× v1 = ~$0.12. Max-permutation test ~$0.00 (CPU-only).

**Honest current claim (if pressed to write the paper today):**
"Forced-advice paired-rollout data on frozen Qwen2.5-7B shows 28% draft accuracy and 27.2% final accuracy (advice is ~neutral at scale), with 6% R→W over-advising and 5.2% W→R repair. A linear probe on the residual stream (layer 1, LR+L2) achieves AUROC 0.745 [0.601, 0.852] for predicting R→W among correct drafts. HOWEVER: (a) the probe signal at `prompt_last` ≈ `gen_last` at layer 1 suggests the signal lives in the question tokens, not the actor's dynamic state; (b) max-test correction over the 116-config search reduces significance substantially; (c) n=70 with 15 positives is near the noise floor. The advisability-as-latent-state thesis is NOT supported by this pilot. The weaker claim — 'a 1-layer problem-feature classifier predicts when forced refinement damages correct drafts' — survives and is still useful (preemptive gating saves compute) but is not the paper I set out to write."

### Iter 11 — Gate results + council decision (2026-04-20, late night)

**Gate results on v1 n=250:**

| Gate | Acc | 95% CI | Advice% | NetReg |
|------|----:|--------|--------:|-------:|
| oracle | **0.332** | [0.276, 0.392] | 5.2% | +0.052 |
| probe-cascade prompt_last L1 | 0.292 | [0.236, 0.352] | 8.0% | +0.012 |
| random @ oracle rate | 0.288 | [0.232, 0.348] | 4.8% | +0.008 |
| logprob-mean matched | 0.284 | [0.228, 0.344] | 5.2% | +0.004 |
| never-advise | 0.280 | [0.224, 0.336] | 0% | 0 |
| always-advise | 0.272 | [0.216, 0.328] | 100% | −0.008 |

**Δ(probe − best logprob) = +0.008. Within bootstrap noise.** The "probe > surface" falsifier does not cleanly pass.

**Council deliberation (Codex + Gemini; Claude unavailable):**

Both converge on: **kill v2 immediately; launch within-problem K-draft + cross-actor experiment.** Reasoning:

- Codex math correction: paired-difference SE for gate-vs-gate at n=500, q=0.25-0.35 discordance → 2SE ≈ 0.045-0.053 (not 0.042). v2 would only matter if true effect is ≥5pp, pilot doesn't support.
- Codex verdict: "A is bad bet (only CI shrinkage); B is highest expected information gain; C dominated by B; D is fallback not experiment."
- Gemini: "Qwen-7B advising Qwen-7B is likely collapsing into mode-seeking identity — you lack the asymmetry of competence required for a meaningful abstention signal." Prescribes **cross-actor (72B advisor → 7B actor)** to break symmetry.
- Gemini: at K=4 drafts, p_correct ≈ 0.29, ~74% of problems will have mixed outcomes → even n=50 gives ~37 informative problems.

**Why K-draft is the decisive A-vs-B test.** Question-only probe predicts all K drafts of a problem identically. Actor-state probe distinguishes which drafts of the same question become R→W versus survive. If Probe-R generalizes to within-problem draft variation, the actor-state claim lives. If not, it's dead.

**Action.** Killing v2. Launching:
1. **K-draft cross-actor collection**: n=50 problems × K=4 drafts at temp=0.6; actor = Qwen-7B transformers (with hidden states); advisor = Qwen-72B-AWQ vLLM. Breaks self-correction symmetry AND isolates question confound in one experiment.
2. **Full-pipeline max-permutation** on v1 (cheap, CPU).

**Budget.** K-draft: ~$0.02. Still well under $10.

### Iter 12 — Infrastructure setback: B200 preempted (2026-04-20, ~04:05 UTC)

**Incident.** Scheduled wake-up triggered ~30 min after K-draft launch. SSH failed with "kex_exchange_identification: read: Connection reset by peer." `flow status capr-diag` shows "paused" with `started_at: null`. `flow health` says "SSH service is still starting up" — indicating a fresh VM, not the one we had.

**Likely cause.** My bid was $0.10, B200 listing price $0.10, win price also $0.10. At the margin — any clearing price increase preempts. Some other bidder likely pushed clearing above $0.10.

**What was running when preempted:**
- K-draft run (~30 min elapsed, probably 30-60 rollouts of 200 done)
- Permutation test on v1 (~70 min elapsed, probably ~50% done)
- Background jobs all terminate on VM destruction

**What's at risk:**
- `/workspace/rae/runs/advis_v1/*` — records, hidden npz, probes summary, gates results. In rae-pilot container's overlay FS → likely GONE.
- `/home/ubuntu/reflective-advisor-evolution/` — source code, records paths. On VM disk → likely GONE.
- `/home/ubuntu/capr-cache/hf/` — Qwen-7B + Qwen-72B-AWQ weights. **MIGHT be on persistent volume** — need to check.

**What's safe:**
- All source code + RESEARCH_LOG.md local at `/Users/duy/Documents/build/dc/reflective-advisor-evolution/`.
- All key numbers from v1 captured in RESEARCH_LOG.md (probe AUROCs, gate accuracies, transition counts).
- MATH data at `decomposition-mve/phase0/actor_eval_k64.jsonl`.

**Action.**
1. Raised bid $0.10 → $0.25 to guarantee allocation.
2. Waiting on SSH to come up (monitor armed).
3. Once up: check if `/home/ubuntu/capr-cache/hf` survived. If yes: re-sync code from local, re-run K-draft from scratch (~2 hours). If no: re-download Qwen-7B (+~10 min), plus re-sync, plus re-run (total ~2.5 hours).

**Contingency: Option D (writeup mode).** If we've truly lost data and can't re-allocate in reasonable time, the honest fallback is writing up v1 as a **negative result paper**:
- Thesis: "Advisability-as-latent-state does NOT hold for same-model MATH forced-advice scaffolds on Qwen-7B at n=250."
- Findings: (a) probe layer-1 is question-feature, not actor-state; (b) probe gate Δ=+0.008 over logprob is within noise; (c) oracle upper bound is only +5.2pp, so advice is nearly neutral at scale — the phenomenon is weaker than Asawa's RuleArena Taxes over-advising would predict.
- Honest but still publishable as a null-diagnosis under the compile-gap umbrella.

### Iter 13 — Recovery + OOM diagnosis (2026-04-20)

**Good news.** New VM provisioned (hostname changed: computeinstance-u00sayapm2b9e4da17, uptime 0min). `/home/ubuntu/reflective-advisor-evolution/` is on a persistent volume — all experimental data intact:
- `runs/advis_v1/records.jsonl` = 250 records ✓
- `runs/advis_v1/probes/`, `gates/`, `surface_features.jsonl` ✓
- `runs/advis_v1/probes_perm/` exists but empty (the permutation test was in progress, never completed)
- `runs/kdraft_v1/records.jsonl` = 200 records ✓ BUT all W→W with empty drafts ⚠️

**Bad news.** K-draft records are **all corrupted**. Every single rollout has `error: OutOfMemoryError: CUDA out of memory...` from trying to allocate 20 MiB with only 15.20 GiB free. The pipeline launched while vLLM (163 GiB) + Qwen-7B (15 GiB) saturated the 183 GiB B200. Same OOM as the very first smoke test.

**My fault.** I approved the launch against the "hope 15GB is enough" plan despite prior OOM evidence. The correct move was to restart vLLM with `--gpu-memory-utilization 0.5` BEFORE launching K-draft.

**Fix.** Recreated `capr-actor` container with `--gpu-memory-utilization 0.5` → vLLM uses ~92GB, leaving ~91GB free for rae-pilot's Qwen-7B. Permutation test also lost (process was killed); records file was never written. Will re-launch both.

**Order of recovery:**
1. ⏳ Wait for vLLM (new config) to be ready.
2. Re-launch K-draft at n=50 × K=4.
3. While K-draft runs: re-launch permutation test on v1 (CPU only, no GPU contention).
4. Once K-draft done: train within-problem probes, compare to question-only.

**Cost of setback.** 1 hour of compute wasted. Still under budget.
