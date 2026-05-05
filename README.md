# Reflective Advisor Evolution

GEPA-evolved Actor → Diagnose → Advise → Revise compound programs that steer a frozen black-box LLM, without ever updating the actor's weights.

The full genealogy of the work — thesis, design, empirical record, methodological audit, open questions — is at [docs/LOG.md](docs/LOG.md).

## Background and motivation

This project sits at the intersection of two recent lines of work on test-time control of language models:

- **Advisor Models** (Asawa et al. 2025, [arXiv:2510.02453](https://arxiv.org/abs/2510.02453)) — train a small advisor model with reinforcement learning to issue per-instance natural-language advice that steers a larger frozen actor. They report +12pp on RuleArena Taxes for GPT-4.1-mini (64.8% → 76.8%) and +71% relative for GPT-5.
- **GEPA** (Agrawal et al. 2025, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)) — evolve textual prompts through reflection on per-example feedback, beating GRPO baselines with up to 35× fewer rollouts. GEPA compiles control policies into *text*; Advisor Models compiles them into *weights*.

The thesis: these are two compilation targets for the same abstract object — a control policy over a frozen executor. The empirical question is:

> How much of a learned advisor policy can be compiled into **text** (evolved prompts) rather than into **weights** (an RL-trained advisor)?

Define `G = J(π_weight) − J(π_text)`. `G ≈ 0` means text is a sufficient representational substrate for advisor policies; `G > 0` characterizes where context-as-control breaks.

The deeper motivation comes from the **Dynamic Cheat Sheet (DCS)** lineage (Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou, [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)): a frozen model maintains a persistent text memory of strategies it writes for itself, retrieves at inference, and reuses. DCS demonstrates this works empirically (≈+27pp on AIME 2024 with Claude 3.5; Game-of-24 from ~10% to ~99% on GPT-4o). What DCS leaves open is *which property* of an entry causes the gain. The advisor compilation question is one way to attack the *when-to-add-an-entry* and *when-not-to-retrieve* sides of that curation problem.

## What's in this repo

A 4-module compound program where the same frozen actor plays four prompt-driven roles. GEPA evolves all four prompts simultaneously, with per-module typed feedback engineered to surface over-advising regressions and abstention failures.

```
              ┌────────────┐
problem ───▶ │ actor_solve│ ─▶ draft
              └────────────┘
                      │
                      ▼
              ┌──────────────────┐
draft + ──▶ │ advisor_diagnose │ ─▶ FAILURE_MODE + EVIDENCE
problem      └──────────────────┘
                      │
                      ▼
              ┌────────────────┐
diagnosis ─▶ │ advisor_advise │ ─▶ concrete hint  | NO_ADVICE
              └────────────────┘
                      │
       ┌──────────────┴───────────────┐
       │                              │
   NO_ADVICE                       advice
       │                              │
       ▼                              ▼
   draft = final              ┌──────────────┐
                              │ actor_revise │ ─▶ final
                              └──────────────┘
```

The `actor_revise` chat layout matches `advisor_models/rule_arena/env.py:_build_student_prompt` verbatim — `[system=actor_solve] [user=question] [assistant=draft] [user=advice + actor_revise]` — so any GEPA delta is attributable to scaffold evolution rather than chat-layout changes. The `compute_score` evaluator has parity tests against Asawa et al.'s published implementation (`tests/test_evaluator.py`).

### Per-module reflective feedback (the GEPA lever)

Per the GEPA paper §5 ("feedback engineering"), the *richness* of textual feedback fed to the reflection LM matters more than scalar reward. The adapter (`src/rae/gepa_adapter.py`) synthesizes per-module diagnostics calling out format failures, missing schema, over-advising regressions, missed repairs, and silence-vs-emit decisions. These are the strings the reflection LM mutates against.

Concrete empirical record, transition-matrix decompositions, the activation-probe campaign, and the methodological audit are in [docs/LOG.md](docs/LOG.md).

## Setup

```bash
# 1. Clone and create a venv
python3 -m venv .venv && source .venv/bin/activate

# 2. Fetch upstream reference repos (advisor-models, gepa, RuleArena)
bash scripts/setup_references.sh

# 3. Install GEPA from the local checkout, then this package and its deps
pip install -e references/gepa
pip install -e .
```

You need an OpenAI-compatible chat-completions endpoint to serve the actor:

```bash
export ACTOR_API_BASE=http://localhost:8000/v1
export ACTOR_API_KEY=sk-...   # any non-empty string for local vLLM
```

Any chat model can serve as the actor; experiments in this repo used Qwen2.5-7B-Instruct and Qwen2.5-72B-Instruct-AWQ via vLLM. The reflection LM is configured by `TOGETHER_API_KEY`, `OPENAI_API_KEY`, or — if neither is set — defaults to the same local endpoint as the actor.

## Running

```bash
# Tests (no GPU required)
PYTHONPATH=src pytest tests/

# RuleArena Taxes — actor-only GEPA (replicates Asawa's static-GEPA baseline)
PYTHONPATH=src python -m rae.run_gepa --arena taxes --variant actor \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 80 --minibatch 3 --acceptance improvement_or_equal

# RuleArena Taxes — full 4-module compound GEPA
PYTHONPATH=src python -m rae.run_gepa --arena taxes --variant full \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 160 --minibatch 3 --module-selector round_robin \
  --acceptance improvement_or_equal

# MATH ZPD (signal-rich subset; pass_rate ∈ [0.15, 0.50])
PYTHONPATH=src python -m rae.run_gepa --arena math_zpd --variant {actor,full} \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 80 --minibatch 3 --acceptance improvement_or_equal
```

## Layout

```
src/rae/
  actor_client.py            # OpenAI-compatible client, retries, parallel-safe
  arenas/
    rule_arena_taxes.py      # loader + ground truth via RuleArena/tax
    math_zpd.py              # signal-rich math subset
  compound_program.py        # 4-module pipeline + NO_ADVICE short-circuit
  evaluator.py               # extract_signed_amount + score_response (Asawa parity)
  gepa_adapter.py            # CompoundProgramAdapter for gepa.optimize
  run_gepa.py                # CLI entry; --variant {actor, full}
  seed_prompts.py            # initial θ for the four modules
  probes.py                  # linear probes on hidden states (activation work)
  collect_advisability.py    # paired-rollout collection for activation probing
  surface_features.py        # surface-feature baselines for the activation probe

tests/
  test_evaluator.py          # regex + score parity to Asawa
  test_compound_program.py   # 4-module flow with mocked actor
  test_gepa_adapter.py       # adapter contract + reflective feedback assertions
  test_collect_advisability.py
  test_probes.py
  test_surface_features.py
  test_gated_eval.py

scripts/
  setup_references.sh             # fetches advisor-models, gepa, RuleArena
  launch_rule_arena_taxes.sh      # parallel-launch helper for the two RuleArena runs

docs/
  LOG.md                          # genealogy: thesis, design, empirical record, audit, open
```

## References

The work in this repo builds directly on:

- Asawa, P., Zhu, A., Zaharia, M., Dimakis, A. G., & Gonzalez, J. E. (2025). *How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models.* arXiv:[2510.02453](https://arxiv.org/abs/2510.02453). Code: [github.com/az1326/advisor-models](https://github.com/az1326/advisor-models).
- Agrawal, L. A. et al. (2025). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* arXiv:[2507.19457](https://arxiv.org/abs/2507.19457). Code: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa).
- Suzgun, M., Yuksekgonul, M., Bianchi, F., Jurafsky, D., & Zou, J. (2025). *Dynamic Cheat Sheet: Test-Time Learning with Adaptive Memory.* arXiv:[2504.07952](https://arxiv.org/abs/2504.07952).

Adjacent / contextual:

- Yuksekgonul, M., Koceja, D., Li, X., Bianchi, F. et al. (2026). *Learning to Discover at Test Time (TTT-Discover).* arXiv:[2601.16175](https://arxiv.org/abs/2601.16175).
- Liang, W., Sun, Y., Nan, S., Li, C., Song, D., & Kawaguchi, K. (2026). *Strategy Executability in Mathematical Reasoning.* arXiv:[2602.22583](https://arxiv.org/abs/2602.22583).
- Wang et al. (2024). *RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios.* arXiv:[2412.08972](https://arxiv.org/abs/2412.08972).

## License

MIT. See `LICENSE`.
