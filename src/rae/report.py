"""One-command results summary for an advisability run.

Reads run_dir/{records.jsonl, surface_features.jsonl, probes/summary.json,
gates/results.jsonl} and writes a compact markdown report suitable for pasting
into a research note or paper-draft appendix.

Usage:
  python -m rae.report --run-dir runs/advis_v1 --output runs/advis_v1/REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_records(run_dir: Path) -> list[dict]:
    with (run_dir / "records.jsonl").open() as f:
        return [json.loads(line) for line in f]


def _transition_table(records: list[dict]) -> dict:
    counts = {"W->W": 0, "W->R": 0, "R->R": 0, "R->W": 0}
    for r in records:
        counts[r["transition"]] = counts.get(r["transition"], 0) + 1
    return counts


def _fmt_probes(probes_summary: dict) -> str:
    rows = []
    rows.append("| Probe | Method | Position | Layer | AUROC | 95% CI |")
    rows.append("|-------|--------|----------|-------|-------|--------|")
    for r in probes_summary["best_per_config"]:
        rows.append(
            f"| {r['probe_name']} | {r['method']} | {r['position']} | {r['layer']} "
            f"| {r['auroc']:.3f} | [{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}] |"
        )
    return "\n".join(rows)


def _fmt_gates(gates: list[dict]) -> str:
    rows = []
    rows.append(
        "| Gate | Accuracy | 95% CI | Advice rate | Net reg | W→R | R→W | R→R | W→W |"
    )
    rows.append(
        "|------|---------:|--------|------------:|--------:|---:|---:|---:|---:|"
    )
    for r in gates:
        rows.append(
            f"| {r['gate_name']} | {r['accuracy']:.3f} "
            f"| [{r['accuracy_ci_lo']:.3f}, {r['accuracy_ci_hi']:.3f}] "
            f"| {r['advice_rate']:.3f} | {r['net_reg']:+.3f} "
            f"| {r['n_WR']} | {r['n_RW']} | {r['n_RR']} | {r['n_WW']} |"
        )
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or (args.run_dir / "REPORT.md")

    records = _load_records(args.run_dir)
    n = len(records)
    draft_acc = sum(r["draft_correct"] for r in records) / n
    final_acc = sum(r["final_correct"] for r in records) / n
    counts = _transition_table(records)
    net_reg = (counts["W->R"] - counts["R->W"]) / n

    lines: list[str] = []
    lines.append(f"# Advisability Probe Run — {args.run_dir.name}")
    lines.append("")
    lines.append(f"**n_problems** = {n}")
    lines.append("")
    lines.append("## Paired rollout summary")
    lines.append("")
    lines.append(f"- draft_acc = **{draft_acc:.3f}**")
    lines.append(f"- final_acc = **{final_acc:.3f}** (forced-advice)")
    lines.append(f"- net regularization = (W→R − R→W)/n = **{net_reg:+.3f}**")
    lines.append("")
    lines.append(
        f"Transitions: W→W={counts['W->W']}, W→R={counts['W->R']}, "
        f"R→R={counts['R->R']}, R→W={counts['R->W']}"
    )
    lines.append("")

    # Probes.
    probes_summary_path = args.run_dir / "probes" / "summary.json"
    if probes_summary_path.exists():
        probes_summary = json.loads(probes_summary_path.read_text())
        lines.append("## Probes (best layer per config)")
        lines.append("")
        lines.append(_fmt_probes(probes_summary))
        lines.append("")

    # Gates.
    gates_path = args.run_dir / "gates" / "results.jsonl"
    if gates_path.exists():
        gates = []
        with gates_path.open() as f:
            for line in f:
                gates.append(json.loads(line))
        lines.append("## Counterfactual gate comparison")
        lines.append("")
        lines.append(_fmt_gates(gates))
        lines.append("")

        # Probe-vs-logprob headline.
        probe_gates = [g for g in gates if g["gate_name"].startswith("probe-cascade")]
        logprob_gates = [g for g in gates if g["gate_name"].startswith("logprob-")]
        if probe_gates and logprob_gates:
            best_probe = max(probe_gates, key=lambda g: g["accuracy"])
            best_lp = max(logprob_gates, key=lambda g: g["accuracy"])
            lines.append("### Headline: probe vs. logprob")
            lines.append("")
            lines.append(
                f"- Best probe-cascade: **{best_probe['accuracy']:.3f}** "
                f"[{best_probe['accuracy_ci_lo']:.3f}, {best_probe['accuracy_ci_hi']:.3f}] "
                f"({best_probe['gate_name']})"
            )
            lines.append(
                f"- Best logprob gate: **{best_lp['accuracy']:.3f}** "
                f"[{best_lp['accuracy_ci_lo']:.3f}, {best_lp['accuracy_ci_hi']:.3f}] "
                f"({best_lp['gate_name']})"
            )
            delta = best_probe["accuracy"] - best_lp["accuracy"]
            lines.append(f"- Δ(probe − logprob) = **{delta:+.3f}**")
            lines.append("")

    # Notes on confounds found.
    lines.append("## Key caveats")
    lines.append("")
    lines.append(
        "- Layer-1 best: prompt_last ≈ gen_last at layer 1 (0.743 vs 0.745 for Probe-R). "
        "Signal is in the question, not the actor's draft-processing state. "
        "This limits the claim to 'preemptive damage classifier' rather than 'latent advisability.'"
    )
    lines.append(
        f"- Minority class sizes are small (W→R={counts['W->R']}, R→W={counts['R->W']}). "
        f"Bootstrap CIs at dim≫n may undercover; scale to 500 problems before headline claims."
    )
    lines.append(
        "- Same-model reflection (Qwen-7B advising Qwen-7B) may encode a self-correction confound. "
        "Cross-model test (Qwen-72B-AWQ advisor) is a necessary follow-up."
    )
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[report] wrote {out}")
    print("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
