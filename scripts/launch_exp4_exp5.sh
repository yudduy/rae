#!/usr/bin/env bash
# Exp4 (actor-only) + Exp5 (full scaffold) on RuleArena Taxes, strong actor.
# Actor: Qwen2.5-72B-Instruct-AWQ via an OpenAI-compatible vLLM endpoint.
# Reflection LM: same endpoint by default; override with TOGETHER_API_KEY or OPENAI_API_KEY.
# Parallel launch — shared vLLM batches the requests.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PY="$(pwd)/.venv/bin/python"
export PYTHONPATH=src

# Exp4: actor-only GEPA baseline (direct analog of Asawa et al. gepa_rule_arena.py)
nohup $VENV_PY -u -m rae.run_gepa \
  --arena taxes --variant actor \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 80 --minibatch 3 \
  --module-selector all --acceptance improvement_or_equal \
  --max-workers 4 --seed 0 \
  --run-dir "$(pwd)/runs/exp4_taxes_actor_72b" \
  > /tmp/rae_exp4.log 2>&1 &
echo "exp4 PID=$!"

# Exp5: GEPA-Advisor full 4-module scaffold — the empty-cell claim
nohup $VENV_PY -u -m rae.run_gepa \
  --arena taxes --variant full \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 160 --minibatch 3 \
  --module-selector round_robin --acceptance improvement_or_equal \
  --max-workers 4 --seed 0 \
  --run-dir "$(pwd)/runs/exp5_taxes_full_72b" \
  > /tmp/rae_exp5.log 2>&1 &
echo "exp5 PID=$!"
