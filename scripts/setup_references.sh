#!/usr/bin/env bash
# Fetch the three upstream reference repos this project pins against.
#
# - advisor-models (Asawa et al., 2510.02453) — RuleArena Taxes data + scoring;
#   our compound program mirrors their _build_student_prompt chat layout
#   verbatim and our evaluator has parity tests against their compute_score.
# - gepa (Agrawal et al., 2507.19457) — pip-installable; provides the
#   reflection engine our CompoundProgramAdapter integrates with.
# - RuleArena (Wang et al., 2412.08972) — the underlying tax-rule arena
#   advisor-models builds on.
#
# These are read-only; do not modify in place.

set -euo pipefail

REF_DIR="$(cd "$(dirname "$0")/.." && pwd)/references"
mkdir -p "$REF_DIR"
cd "$REF_DIR"

clone_or_update () {
  local url="$1"
  local dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "[setup_references] $dest exists, fetching latest..."
    (cd "$dest" && git fetch --quiet)
  else
    echo "[setup_references] Cloning $url -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

clone_or_update https://github.com/az1326/advisor-models.git advisor-models
clone_or_update https://github.com/gepa-ai/gepa.git gepa
clone_or_update https://github.com/skyriver-2000/RuleArena.git RuleArena

echo
echo "[setup_references] Done. Install GEPA into your venv with:"
echo "    pip install -e $REF_DIR/gepa"
