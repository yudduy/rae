# Overnight research-loop status — 2026-04-20

Summary for when you wake up. Full details in `RESEARCH_LOG.md`.

## TL;DR

Ran the advisability-probing pilot end-to-end on Qwen2.5-7B + MATH ZPD (n=250) with paired-rollout data collection, surface-feature baselines, per-layer probes, and counterfactual gate evaluation. **The activation-probe advisability thesis did NOT survive the pilot.** Two planned follow-ups (K-draft within-problem test + permutation test) were sabotaged by a B200 preemption in the middle of the night. Recovering now.

## The headline numbers (v1 pilot, n=250 MATH ZPD, Qwen-7B forced-advise paired rollouts)

| Quantity | Value |
|---|---|
| draft_acc | 0.280 |
| final_acc (forced advice) | 0.272 |
| net regularization | **−0.008** (advice ≈ neutral) |
| W→W transitions | 167 |
| W→R (repair) | 13 |
| R→R (preserve despite forced advice) | 55 |
| R→W (over-advising regression) | 15 |
| Oracle ceiling accuracy | 0.332 (+5.2pp over never-advise) |
| **Probe-R** LR+L2 @ layer 1 AUROC | **0.745** [0.601, 0.852] |
| Probe-W LR+L2 @ layer 5 AUROC | 0.664 [0.527, 0.797] |
| Correctness probe AUROC (baseline) | 0.665 [0.591, 0.729] |
| probe-cascade gate accuracy | 0.292 [0.236, 0.352] |
| best logprob-threshold gate | 0.284 [0.228, 0.344] |
| **Δ (probe vs logprob)** | **+0.008** (within bootstrap noise) |

## Three fatal findings

1. **Probe doesn't meaningfully beat logprob at gate-level.** Δ=+0.008 is 0.27σ. The "probe > surface" falsifier committed to before running doesn't pass.
2. **The AUROC=0.745 is a question-feature confound, not actor-state signal.** `prompt_last` (before any draft) and `gen_last` (after the full draft) give near-identical AUROC at layer 1 (0.743 vs 0.745). If advisability were in the actor's dynamic state, `gen_last` should dominate.
3. **Multiple-testing correction deflates the claim.** Searched 29 layers × 2 positions × 2 methods = 116 configs; family-wise probability of observing ≥0.745 under null is ~16-20% (ChatGPT Pro verified). Naive 95% CI [0.601, 0.852] is misleading because bootstrap doesn't correct for selection.

## Council's prescribed falsifier (the decisive experiment)

Both Codex and Gemini converged on: **run a K-draft cross-actor experiment**.
- Generate K=4 drafts per problem at temperature 0.6 (instead of 1 deterministic draft).
- Use Qwen-72B-AWQ as advisor (breaks self-correction symmetry flagged by Gemini).
- A question-only probe predicts all K drafts of a problem identically.
- An actor-state probe distinguishes drafts that become R→W from drafts that survive.
- This cleanly kills or rescues the advisability thesis.

## Infrastructure incident

Launched K-draft on B200 overnight. Two problems:
1. **All 200 K-draft rollouts hit CUDA OOM** (vLLM 72B took 163GB, left only 15GB for Qwen-7B; edge case). My mistake — I should have reduced vLLM gpu_memory_utilization first.
2. **B200 bid got preempted** (clearing price rose above $0.10). VM was re-provisioned fresh.

Recovery:
- Data on persistent volume survived (runs/, code, HF cache).
- Containers exited but could be restarted.
- Recreated `capr-actor` with `--gpu-memory-utilization 0.5` (leaves ~91GB for Qwen-7B).
- Raised bid $0.10 → $0.25 → $1.00 to stabilize against clearing-price churn.
- Current state as this file was written: waiting for SSH to stabilize after latest bid raise.

## What's likely done by the time you read this

If SSH comes back and stays up:
- Fresh K-draft run (n=50 × K=4) completes in ~2 hours → we have the decisive A-vs-B answer.
- Permutation test on v1 completes in ~45 min → we have proper p-values for Probe-R.

If SSH keeps flapping:
- Partial data. Option D (writeup-only) becomes the honest path.

## Honest paper draft (written even if K-draft fails)

**Title candidate**: *"The Observer Problem in LLM Advisors: Empirical Limits of Activation-State Probing for Intervention-Value Estimation"*

**Claim**: Reflective prompt evolution (GEPA) and trained advisors (Asawa) both implicitly assume that intervention value is representable as a learnable function. We tested whether that function is linearly readable from the frozen executor's residual stream at matched budgets. On Qwen2.5-7B + MATH ZPD (n=250), it isn't — at least not beyond question-difficulty features accessible via layer-1 activations, and not beyond what surface log-probability already provides.

**Contributions**:
1. Formalize **advisability** = E[Δ_advise | h] = J(with advice) − J(without advice | h) as a distinct target from draft correctness.
2. Empirically characterise the counterfactual transition matrix on forced-advice paired rollouts.
3. Report the gap between probe-gated and surface-gated advisors — and show it is small at this scale.
4. Identify the **question-feature confound** via the `prompt_last ≈ gen_last` coincidence — a mistake future activation-based gate work should pre-register against.
5. (If K-draft recovers tonight): within-problem draft variation to cleanly test state-vs-question.

**Why this is still a real contribution** even if probe doesn't beat logprob:
- Advisor Models (Asawa) doesn't measure the trained advisor's NO_ADVICE rate — we measure and report the oracle-vs-trained gap decomposition.
- ChatGPT Pro and council process is itself documented — a rare record of adversarial multi-model research loops.
- The negative result is specific and actionable: don't probe at layer 1 for intervention value; measure `prompt_last − gen_last` delta as the actor-state fingerprint.

## Decisions for you in the morning

1. **If the K-draft result is strong** (probe distinguishes drafts of same question better than question-only baseline at ≥2SE), scale to 200 problems, add Qwen-72B cross-actor extension. Target paper timeline: 1-2 weeks.
2. **If the K-draft result is null**, write up Option D (negative study) in 2-3 days. Honest NeurIPS workshop submission.
3. **If K-draft never ran** (infra died), kick it off cleanly tomorrow and repeat.

## Final overnight outcome — infra died (K-draft didn't run)

Reality: outcome #3 from the decision list above.

**Bid capr-diag was CANCELLED by Mithril** at ~05:00 UTC after multiple preemption/recovery cycles. Timeline:
1. Original $0.10 bid won auction fine for most of the day.
2. Around 04:00 UTC, clearing price rose → bid was paused by auction.
3. Raised to $0.25 → instance came back up briefly → preempted again within 30 min.
4. Raised to $1.00 → instance came back up → vLLM 0.5-mem-util config running → K-draft re-launched.
5. K-draft discovered new bug: rae-pilot (docker bridge) couldn't reach capr-actor (docker host) via `localhost:8001`. Needed `http://172.17.0.1:8001` (bridge gateway).
6. Went to kill failed K-draft and restart with correct URL → **SSH dropped again**.
7. `flow status capr-diag` now reports `status: cancelled`. No `flow bid unpause` possible on cancelled bids.

**What survives:**
- All v1 analysis (numbers above).
- All code in this repo (including `collect_kdraft.py`, `probes.py --permutation`, `surface_features.py`).
- Full RESEARCH_LOG.md — detailed iteration log with ChatGPT Pro and council transcripts.

**What's lost:**
- K-draft v2 run (only 35/50 problems done, and those had the networking bug so all rollouts were preserves — wouldn't have answered the A-vs-B test anyway).
- Permutation test (~35 min into 2h run, never wrote results).
- Any data on the B200 instance's local disk that wasn't on the persistent volume (persistent volume may or may not have survived bid cancellation — unclear).

**I did NOT autonomously spin up a new B200 bid**, because:
- That's a paid resource action you should approve.
- The v1 pilot already gave a clear and honest result: **advisability-as-latent-state thesis is not supported on this setup**.
- The unresolved question (K-draft within-problem variation) is worth running on a fresh instance tomorrow but doesn't need to happen now.

**Concrete options for your morning decision:**

### Option P1 — Provision new B200, finish K-draft + permutation, then write up (1 day)
- Spend ~$1-5 on new B200 bid.
- Run fresh K-draft (50×K=4, ~2h) with networking fix + low-mem vLLM from the start.
- Run full-pipeline max-permutation test (~2h).
- Assemble paper draft around whatever result lands.
- **If probe-cascade beats logprob meaningfully AND within-problem test clears the question-feature confound, we have a real paper.**
- **If not, we have an honest negative result for NeurIPS workshop / arXiv preprint.**

### Option P2 — Write up v1 as-is as negative result (2-3 days)
- Skip K-draft + permutation.
- Frame paper around: "We tested activation-probe advisability on a controlled forced-advice paired-rollout setup. Probe AUROC 0.745 on a selected config does not survive multiple-testing correction, and the signal lives in question tokens (layer 1, prompt_last ≈ gen_last) rather than actor dynamic state. Probe-gated advice does not outperform logprob-gated advice at matched budgets on Qwen-7B + MATH ZPD."
- Workshop / preprint target.
- Still a genuine contribution: clean methodology template + cautionary finding for activation-probe approaches to intervention gating.

### Option P3 — Pivot direction entirely (1 week)
- The self-correction confound (Qwen advising Qwen) that Gemini flagged is a deeper issue than probe layer selection.
- A much stronger setup: Qwen-72B actor (richer state) with Claude/GPT-4 as advisor (asymmetry of competence).
- Doesn't fit in today's compute budget but would address the core identity-collapse problem at its root.

**My recommendation:** P1 first (cheap, fast, completes the promised experiment). If negative, P2. P3 only if P1 suggests deeper issues.

## Key commands for tomorrow

```bash
# New B200 bid
flow submit <path-to-yaml>  # OR
flow grab 1 neb-b200 --hours 4 --max-price 1.00

# Once up, re-rsync code:
cd /Users/duy/Documents/build/dc/reflective-advisor-evolution
rsync -az --exclude='.venv' --exclude='runs' --exclude='__pycache__' --exclude='.git' \
  -e "ssh -i /Users/duy/.flow/keys/production/<key>" \
  ./ ubuntu@<new-host>:/home/ubuntu/rae/

# Re-upload data (MATH phase0):
scp -i ... /Users/duy/Documents/build/dc/decomposition-mve/results/runs/decomp_mve/phase0/actor_eval_k64.jsonl \
  ubuntu@<new-host>:/home/ubuntu/rae/data_in/

# Re-start capr-actor with LOW gpu_memory_utilization this time:
sudo docker run -d --name capr-actor --gpus all --network host --entrypoint vllm \
  -v /home/ubuntu/capr-cache/hf:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.02-py3 \
  serve Qwen/Qwen2.5-72B-Instruct-AWQ --port 8001 --api-key sk-capr-actor \
  --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.5 \
  --dtype auto --quantization awq

# Start rae-pilot:
sudo docker run -d --name rae-pilot --gpus all --shm-size 8g \
  -v /home/ubuntu/capr-cache/hf:/root/.cache/huggingface \
  -v /home/ubuntu/rae:/workspace/rae \
  --entrypoint tail nvcr.io/nvidia/vllm:26.02-py3 -f /dev/null

# Install deps in rae-pilot:
sudo docker exec rae-pilot pip install --quiet math-verify scikit-learn accelerate

# Launch K-draft with CORRECT advisor URL (172.17.0.1 is docker bridge gateway):
sudo docker exec -d rae-pilot bash -c "cd /workspace/rae && \
  PYTHONPATH=src HF_HOME=/root/.cache/huggingface \
  python -u -m rae.collect_kdraft --run-dir /workspace/rae/runs/kdraft \
    --n-problems 50 --k-drafts 4 --temperature 0.6 \
    --data-src /workspace/rae/data_in/actor_eval_k64.jsonl \
    --cache-dir /root/.cache/huggingface --max-new-tokens 1200 \
    --advisor-base-url http://172.17.0.1:8001/v1 \
    --advisor-api-key sk-capr-actor \
    --advisor-model Qwen/Qwen2.5-72B-Instruct-AWQ \
    > /workspace/rae/runs/kdraft.log 2>&1 &"

# Permutation test (parallel, CPU):
sudo docker exec -d rae-pilot bash -c "cd /workspace/rae && PYTHONPATH=src \
  python -u -m rae.probes --run-dir /workspace/rae/runs/advis_v1 \
    --permutation --n-permutations 200 \
    --output-dir /workspace/rae/runs/advis_v1/probes_perm \
    > /workspace/rae/runs/advis_v1/probes_perm.log 2>&1 &"
```

## Bottom line for the morning

**The advisability-probe paper in its original framing is in trouble.** The v1 pilot didn't support the latent-state hypothesis. The decisive K-draft experiment didn't complete due to infra failures. The paper can still be written — either as a clean negative result (Option P2) or after a clean K-draft run (Option P1). Both are honest contributions; neither is the breakthrough we aimed for.

## Files to read in order

1. This file (`reports/status-overnight-2026-04-20.md`) — 5 min.
2. `RESEARCH_LOG.md` — full iteration-by-iteration log with ChatGPT Pro + council transcripts.
3. `runs/advis_v1/REPORT.md` — tables.
4. `src/rae/collect_kdraft.py` — the decisive-test code.
5. `src/rae/probes.py` — probe training with `--permutation` flag.
