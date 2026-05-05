#!/usr/bin/env bash
# Two parallel RuleArena Taxes runs against a strong actor:
#   1. actor-only GEPA  (replicates Asawa et al.'s static-GEPA baseline)
#   2. full 4-module compound GEPA  (the empty-cell claim)
#
# Actor: Qwen2.5-72B-Instruct-AWQ via an OpenAI-compatible vLLM endpoint.
# Reflection LM: same endpoint by default; override via TOGETHER_API_KEY or
# OPENAI_API_KEY.
# Parallel launch — a shared vLLM batches the requests.

set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PY="$(pwd)/.venv/bin/python"
export PYTHONPATH=src

# 1. Actor-only GEPA baseline.
nohup $VENV_PY -u -m rae.run_gepa \
  --arena taxes --variant actor \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 80 --minibatch 3 \
  --module-selector all --acceptance improvement_or_equal \
  --max-workers 4 --seed 0 \
  --run-dir "$(pwd)/runs/taxes_actor_only" \
  > /tmp/rae_taxes_actor_only.log 2>&1 &
echo "taxes_actor_only PID=$!"

# 2. Full 4-module compound GEPA — actor → diagnose → advise → revise.
nohup $VENV_PY -u -m rae.run_gepa \
  --arena taxes --variant full \
  --train-n 30 --dev-n 15 --holdout-n 15 \
  --budget 160 --minibatch 3 \
  --module-selector round_robin --acceptance improvement_or_equal \
  --max-workers 4 --seed 0 \
  --run-dir "$(pwd)/runs/taxes_full_compound" \
  > /tmp/rae_taxes_full_compound.log 2>&1 &
echo "taxes_full_compound PID=$!"
