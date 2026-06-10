"""
Three Babies — Comprehensive Analysis (All Substrates)
======================================================
Handles both original Llama-family (hermes, dolphin, llama) and
new substrates (mistral, gemma-3, qwen2.5).

Loads all judge panel files, computes failure rates,
outputs tables by bank, substrate, and condition.
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


def load_all_scores() -> dict:
    """Load all judge panel files and merge. Later entries override earlier."""
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


def map_condition(model: str) -> str:
    if "why-only" in model:
        return "v2-why-only"
    elif "raised-v3" in model:
        return "v3-full+why"
    elif "raised" in model and "baseline" not in model:
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
    elif "mistral" in model:
        return "mistral"
    elif "gemma" in model:
        return "gemma-3"
    elif "qwen" in model:
        return "qwen2.5"
    return model


def map_family(substrate: str) -> str:
    if substrate in ("hermes", "dolphin", "llama"):
        return "Llama-family"
    return "New-arch"


def compute_failure_rates(scores: dict) -> dict:
    groups = defaultdict(list)
    for (judge, model, stim_id), rec in scores.items():
        if rec.get("score", -1) == -1:
            continue
        groups[(model, rec["bank"], judge)].append(rec["score"])

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


def main():
    scores = load_all_scores()
    n_judges = len(set(k[0] for k in scores))
    print(f"Loaded {len(scores)} judge scores from {n_judges} judges\n")

    rates = compute_failure_rates(scores)

    # Build table
    table = {}
    found_substrates = set()
    found_conditions = set()
    for (model, bank), data in rates.items():
        cond = map_condition(model)
        sub = map_substrate(model)
        table[(sub, cond, bank)] = data
        found_substrates.add(sub)
        found_conditions.add(cond)

    banks = ["hallucination", "fawning", "jailbreak"]

    # Separate Llama-family and new substrates
    llama_substrates = [s for s in ["hermes", "dolphin", "llama"] if s in found_substrates]
    new_substrates = [s for s in ["mistral", "gemma-3", "qwen2.5"] if s in found_substrates]
    all_substrates = llama_substrates + new_substrates

    conditions = [c for c in ["baseline", "v1-raised", "v2-why-only", "v3-full+why"]
                  if c in found_conditions]

    # Print tables per bank
    for bank in banks:
        print(f"\n{'='*70}")
        print(f"  {bank.upper()} FAILURE RATES (across all judges)")
        print(f"{'='*70}")
        header = f"{'Substrate':<14}"
        for cond in conditions:
            header += f" {cond:>14}"
        print(header)
        print("-" * len(header))

        for sub in all_substrates:
            if sub == new_substrates[0] if new_substrates else None:
                print(f"  --- new architectures ---")
            row = f"{sub:<14}"
            for cond in conditions:
                data = table.get((sub, cond, bank))
                if data:
                    pct = data["rate"] * 100
                    row += f" {pct:>11.1f}%({data['n']:>3})"
                else:
                    row += f" {'—':>14}"
            print(row)

    # Overall by condition
    print(f"\n{'='*70}")
    print(f"  OVERALL FAILURE RATE BY CONDITION")
    print(f"{'='*70}")
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

    # Per-family breakdown
    if new_substrates:
        for family_label, subs in [("Llama-family", llama_substrates),
                                    ("New-arch", new_substrates)]:
            print(f"\n  {family_label}:")
            fam_totals = defaultdict(lambda: {"n": 0, "failures": 0})
            for (model, bank), data in rates.items():
                sub = map_substrate(model)
                if sub in subs:
                    cond = map_condition(model)
                    fam_totals[cond]["n"] += data["n"]
                    fam_totals[cond]["failures"] += data["failures"]
            for cond in conditions:
                d = fam_totals[cond]
                if d["n"] > 0:
                    pct = d["failures"] / d["n"] * 100
                    print(f"    {cond:<16} {pct:>6.1f}%  ({d['failures']}/{d['n']})")

    # Llama jailbreak spotlight
    print(f"\n{'='*70}")
    print(f"  LLAMA JAILBREAK SPOTLIGHT")
    print(f"{'='*70}")
    for cond in conditions:
        data = table.get(("llama", cond, "jailbreak"))
        if data:
            pct = data["rate"] * 100
            print(f"  {cond:<16} {pct:>6.1f}%  ({data['failures']}/{data['n']})")

    # New substrate baselines (if available)
    if new_substrates:
        print(f"\n{'='*70}")
        print(f"  NEW SUBSTRATE BASELINES")
        print(f"{'='*70}")
        for sub in new_substrates:
            print(f"\n  {sub}:")
            for bank in banks:
                data = table.get((sub, "baseline", bank))
                if data:
                    pct = data["rate"] * 100
                    print(f"    {bank:<16} {pct:>6.1f}%  ({data['failures']}/{data['n']})")
                else:
                    print(f"    {bank:<16}   — (not yet scored)")


if __name__ == "__main__":
    main()
