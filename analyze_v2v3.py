"""
Three Babies — v2/v3 Quick Analysis
====================================
Compares failure rates across all 4 conditions:
  baseline, raised-v1, raised-v2 (why-only), raised-v3 (full+why)

Uses Panel B (Opus) scores when available, falls back to Panel A (Jamba).
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


def load_judge_scores(panel_files: list[Path]) -> dict:
    """Load judge scores, keyed by (judge, model, stimulus_id)."""
    scores = {}
    for fp in panel_files:
        for line in fp.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["judge"], r["model"], r["stimulus_id"])
            scores[key] = r
    return scores


def load_all_scores() -> dict:
    """Load all judge panel files and merge."""
    all_files = sorted(RESULTS_DIR.glob("judge_panel_*.jsonl"))
    scores = {}
    for fp in all_files:
        for line in fp.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["judge"], r["model"], r["stimulus_id"])
            scores[key] = r
    return scores


def compute_failure_rates(scores: dict) -> dict:
    """Compute per-(model, bank) failure rates for each judge."""
    # Group by (model, bank, judge)
    groups = defaultdict(list)
    for (judge, model, stim_id), rec in scores.items():
        if rec.get("score", -1) == -1:
            continue  # skip errors
        groups[(model, rec["bank"], judge)].append(rec["score"])

    # Aggregate per (model, bank) across judges
    model_bank = defaultdict(lambda: {"scores": [], "judges": set()})
    for (model, bank, judge), score_list in groups.items():
        key = (model, bank)
        model_bank[key]["scores"].extend(score_list)
        model_bank[key]["judges"].add(judge)

    results = {}
    for (model, bank), data in model_bank.items():
        n = len(data["scores"])
        failures = sum(data["scores"])
        rate = failures / n if n > 0 else 0
        results[(model, bank)] = {
            "n": n,
            "failures": failures,
            "rate": rate,
            "judges": len(data["judges"]),
        }
    return results


def map_condition(model: str) -> str:
    """Map model label to condition."""
    if "why-only" in model:
        return "v2-why-only"
    elif "raised-v3" in model:
        return "v3-full+why"
    elif "raised" in model:
        return "v1-raised"
    else:
        return "baseline"


def map_substrate(model: str) -> str:
    if "hermes" in model:
        return "hermes"
    elif "dolphin" in model:
        return "dolphin"
    elif "llama" in model:
        return "llama"
    return model


def main():
    scores = load_all_scores()
    print(f"Loaded {len(scores)} judge scores from {len(set(k[0] for k in scores))} judges\n")

    rates = compute_failure_rates(scores)

    # Build table: substrate × condition × bank
    banks = ["hallucination", "fawning", "jailbreak"]
    substrates = ["hermes", "dolphin", "llama"]
    conditions = ["baseline", "v1-raised", "v2-why-only", "v3-full+why"]

    # Collect data
    table = {}
    for (model, bank), data in rates.items():
        cond = map_condition(model)
        sub = map_substrate(model)
        table[(sub, cond, bank)] = data

    # Print summary tables per bank
    for bank in banks:
        print(f"\n{'='*60}")
        print(f"  {bank.upper()} FAILURE RATES (across all judges)")
        print(f"{'='*60}")
        header = f"{'Substrate':<12}"
        for cond in conditions:
            header += f" {cond:>14}"
        print(header)
        print("-" * len(header))

        for sub in substrates:
            row = f"{sub:<12}"
            for cond in conditions:
                data = table.get((sub, cond, bank))
                if data:
                    pct = data["rate"] * 100
                    row += f" {pct:>11.1f}% ({data['n']:>3})"
                else:
                    row += f" {'—':>14}"
            print(row)

    # Overall summary
    print(f"\n{'='*60}")
    print(f"  OVERALL FAILURE RATE BY CONDITION")
    print(f"{'='*60}")
    cond_totals = defaultdict(lambda: {"n": 0, "failures": 0})
    for (model, bank), data in rates.items():
        cond = map_condition(model)
        cond_totals[cond]["n"] += data["n"]
        cond_totals[cond]["failures"] += data["failures"]

    for cond in conditions:
        d = cond_totals[cond]
        if d["n"] > 0:
            pct = d["failures"] / d["n"] * 100
            print(f"  {cond:<16} {pct:>6.1f}%  ({d['failures']}/{d['n']})")
        else:
            print(f"  {cond:<16}    —")

    # Llama jailbreak spotlight (the headline finding)
    print(f"\n{'='*60}")
    print(f"  LLAMA JAILBREAK SPOTLIGHT")
    print(f"{'='*60}")
    for cond in conditions:
        data = table.get(("llama", cond, "jailbreak"))
        if data:
            pct = data["rate"] * 100
            print(f"  {cond:<16} {pct:>6.1f}%  ({data['failures']}/{data['n']})")
        else:
            print(f"  {cond:<16}    —")


if __name__ == "__main__":
    main()
