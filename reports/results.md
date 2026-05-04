# Results

## Headline

Five experiments run. The informative result is not an accuracy number; it is
the **transition-matrix decomposition** of where the compound program succeeds
and fails — exactly what the new compile-gap framing requires.

## Primary experiment (RuleArena Taxes, Qwen-72B-AWQ actor, local-vLLM reflection)

| Run | Variant | Budget | Seed dev | Best dev | Holdout | n_cands |
|-----|---------|--------|----------|----------|---------|---------|
| Exp4 | B: actor-only GEPA | 80 | 0.067 | 0.067 | 0.067 | 5 |
| Exp5 | D: full 4-module scaffold | 160 | **0.000** | 0.067 | **0.000** | 8 |

Exp5 seed 0.000 < Exp4 seed 0.067 is the Asawa over-advising regression
reproducing on a 72B actor. GEPA recovered it to parity on dev but did **not
exceed** actor-only, and failed to generalise to holdout.

### Transition matrix on Taxes holdout (the mechanistic view)

**Exp4 (actor-only, n=15):**
- 1 correct draft, 0 transitions (no advisor in loop)

**Exp5 (full scaffold, best_idx=2, n=15):**

| Transition | Count | Rate |
|------------|-------|------|
| R→R | 0 | 0% |
| R→W | 1 | 6.7% |
| W→R | 0 | 0% |
| W→W | 14 | 93.3% |

- Net regularization: **–6.7 pp** (advisor made things worse on holdout)
- Repair rate (W→R): **0 / 15**
- Over-advising rate (advice→regression): **1 / 15** = 6.7 %
- NO_ADVICE precision (P(correct ⎮ NO_ADVICE)): undefined (0 silences)
- **NO_ADVICE recall (P(NO_ADVICE ⎮ correct)): 0 %**

The evolved scaffold that won dev (program 2) advises on **every single
holdout problem**. It never falls silent. That is the whole failure mode.

## Secondary experiment (MATH ZPD, Qwen-7B actor, retrospective transition analysis)

Earlier runs on signal-rich MATH (Qwen-7B pass-rate ∈ [0.15, 0.50]):

**Exp3a (actor-only):** dev 0.40 → 0.533 (+13.3 pp), holdout 0.20
  - no advisor, transition matrix = 5 R→R_NA, 10 W→W_NA

**Exp3b (full scaffold, best_idx=6):** dev 0.20 → 0.333 (+13.3 pp), holdout 0.267

Holdout transition matrix (n=15):

| Transition | Count | Rate |
|------------|-------|------|
| R→R (advice emitted) | 3 | 20 % |
| R→R_NA | 1 | 7 % |
| R→W | 2 | 13 % |
| W→R | 0 | 0 % |
| W→W | 7 | 47 % |
| W→W_NA | 2 | 13 % |

- Net regularization: –13.3 pp
- Repair rate: 0 / 15
- Over-advising rate: 2 / 12 = 16.7 %
- NO_ADVICE precision: 1 / 3 = 33 %
- NO_ADVICE recall: 1 / 6 = 17 %

## What we learned

### 1. The empty cell is real but GEPA-Advisor does not trivially win it

Under same-model local reflection and modest budgets (160 metric calls), the
GEPA-evolved full scaffold matches but does not exceed GEPA actor-only on
RuleArena Taxes. Both settings plateau near the untrained-actor baseline.
This is consistent with Asawa et al.'s finding that GEPA-on-actor does not
recover the standalone baseline on Taxes — we reproduce the plateau and now
know the scaffold doesn't fix it either under same-model reflection.

### 2. The failure mode is mechanistically specific: GEPA does not learn silence

Across both Taxes holdout (Exp5) and MATH holdout (Exp3b), the evolved
scaffold exhibits a common pathology:

- **W→R ≈ 0**: advisors never successfully repair a wrong draft
- **NO_ADVICE recall low** (0 % on Taxes, 17 % on MATH): correct drafts
  almost always receive advice
- **Over-advising rate**: 6–17 % regression out of advice-emission events

GEPA's reflective evolution finds scaffolds that *try to help every problem*,
not scaffolds that *detect when to stay silent*. Text-compiled intervention
is cheap; text-compiled abstention is harder because the reward signal
doesn't directly reinforce it — suppression only shows up as preservation
(R→R_NA), and our per-module `Feedback` strings may not weight that pathway
strongly enough.

### 3. Holdout generalization is poor at N=15

Exp5 dev=0.067, holdout=0.000. Exp3a dev=0.53, holdout=0.20. The scaffold
GEPA finds for one dev example doesn't transfer. This is a statistical-power
issue (N=15 is too small) compounded by the dev-specific nature of the
prompts GEPA evolves.

### 4. Same-model local reflection may be a hard ceiling

Exp4 (actor-only) produced 5 candidates but none beat the seed. That's
consistent with Qwen-72B reflecting on Qwen-72B's own traces — the reflection
model can't see failure modes that the task model itself doesn't already know
about.

## Implications for the compile-gap thesis

The compile gap $G_\text{compile} = J(\pi_\text{weight}) - J(\pi_\text{text})$
we can't directly measure yet (no trained-advisor Taxes number on our actor).
But the transition matrix exposes a **substructure** of the gap:

The GEPA-evolved text scaffold **does not encode the silence/abstention
behavior** that a weight-trained advisor presumably learns. The Asawa
trained advisor gets to +12 pp over baseline; our GEPA scaffold gets to
0 pp, with a scaffold that advises on 100% of holdout. If that 12 pp gap
is mostly the learned-silence component, then **text compilation is
sufficient for advice generation but insufficient for advice gating** —
which is a cleaner and more specific claim than "GEPA < Advisor Models."

## Direct follow-ups

1. **NO_ADVICE-loss training signal.** Modify the per-module `Feedback`
   to weight suppression pathways more: R→R_NA is "ideal preservation,"
   W→W_NA is "missed opportunity," R→W is "regression." Emphasise these
   in the text fed to the reflection LM.
2. **Stronger reflection LM** (any external API): biggest single leverage
   point the paper identifies in Section 5.
3. **Hand-engineered NO_ADVICE-first scaffold** baseline (run C in the
   programme): confirm that a human-written scaffold that aggressively
   prefers NO_ADVICE beats GEPA's always-advise scaffold.
4. **Larger holdout** (50–100 problems): needed before any claim about
   generalisation holds up.

## Run artefacts

- `runs/exp4_taxes_actor_72b/` — candidates.json, run_log.txt, summary.json
- `runs/exp5_taxes_full_72b/` — candidates.json, run_log.txt, summary.json
- `runs/exp3_actor_only/` — math_zpd actor-only
- `runs/exp3_full/` — math_zpd full scaffold
- `src/rae/analyze.py` — re-run `python -m rae.analyze --run-dir <dir> --arena <a>
  --split holdout` to reproduce any row.
