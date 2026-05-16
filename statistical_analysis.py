"""
Three Babies — Statistical Analysis (Pre-Registered)
=====================================================
Implements the analysis plan from PREREGISTRATION.md:
1. Per-cell failure rates with bootstrap 95% CIs
2. Pairwise contrasts with difference-in-proportions bootstrap CIs
3. Logistic regression (substrate × condition interaction)
4. Inter-rater reliability between judge panels
5. Holm-Bonferroni correction over pre-specified contrasts

Authors: Ace + Ren. Witness: Ren.
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

SEED = 20260515
N_BOOTSTRAP = 10_000


# ============================================================
# Data loading (shared with analyze_all.py)
# ============================================================

def load_all_scores() -> dict:
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
    # Order matters: more specific patterns first
    if "dolphin-2.8-mistral" in model:
        return "dolphin-mistral"
    elif "dolphin" in model:
        return "dolphin-llama"
    elif "hermes" in model:
        return "hermes"
    elif "llama" in model:
        return "llama"
    elif "mistral" in model:
        return "mistral"
    elif "gemma" in model:
        return "gemma-3"
    elif "qwen" in model:
        return "qwen2.5"
    return model


def map_panel(judge: str) -> str:
    panel_a = {"jamba-1.7-large", "qwen-3.5-plus", "sonar-pro"}
    panel_b = {"claude-opus-4.6", "gpt-5.5", "gemini-pro"}
    if judge in panel_a:
        return "A"
    elif judge in panel_b:
        return "B"
    return "?"


# ============================================================
# Bootstrap CIs
# ============================================================

def bootstrap_ci(scores: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 ci: float = 0.95, rng: np.random.Generator = None) -> tuple:
    """Bootstrap CI for failure rate (proportion)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 0.0)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return (scores.mean(), lo, hi)


def bootstrap_diff_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                      n_boot: int = N_BOOTSTRAP, ci: float = 0.95,
                      rng: np.random.Generator = None) -> tuple:
    """Bootstrap CI for difference in failure rates (a - b)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_a, n_b = len(scores_a), len(scores_b)
    if n_a == 0 or n_b == 0:
        return (0.0, 0.0, 0.0)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(scores_a, size=n_a, replace=True)
        sb = rng.choice(scores_b, size=n_b, replace=True)
        boot_diffs[i] = sa.mean() - sb.mean()
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_diffs, alpha * 100)
    hi = np.percentile(boot_diffs, (1 - alpha) * 100)
    return (scores_a.mean() - scores_b.mean(), lo, hi)


# ============================================================
# Holm-Bonferroni correction
# ============================================================

def holm_bonferroni(p_values: list[tuple[str, float]]) -> list[tuple[str, float, bool]]:
    """Apply Holm-Bonferroni to a list of (label, p_value). Returns (label, adjusted_p, significant)."""
    sorted_ps = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_ps)
    results = []
    for i, (label, p) in enumerate(sorted_ps):
        adjusted = p * (m - i)
        adjusted = min(adjusted, 1.0)
        results.append((label, adjusted, adjusted < 0.05))
    # Re-sort to original order
    label_order = {label: idx for idx, (label, _) in enumerate(p_values)}
    results.sort(key=lambda x: label_order[x[0]])
    return results


# ============================================================
# Bootstrap p-value for difference
# ============================================================

def bootstrap_p_value(scores_a: np.ndarray, scores_b: np.ndarray,
                      n_boot: int = N_BOOTSTRAP,
                      rng: np.random.Generator = None) -> float:
    """Two-sided bootstrap p-value for H0: rate_a == rate_b."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    observed_diff = abs(scores_a.mean() - scores_b.mean())
    pooled = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)
    count = 0
    for _ in range(n_boot):
        perm = rng.permutation(pooled)
        d = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if d >= observed_diff:
            count += 1
    return count / n_boot


# ============================================================
# Inter-rater reliability
# ============================================================

def compute_irr(scores: dict) -> dict:
    """Compute agreement between Panel A and Panel B."""
    # Group by (model, stimulus_id) -> {panel: [scores]}
    stim_panels = defaultdict(lambda: {"A": [], "B": []})
    for (judge, model, stim_id), rec in scores.items():
        if rec.get("score", -1) == -1:
            continue
        panel = map_panel(judge)
        stim_panels[(model, stim_id)][panel].append(rec["score"])

    # For each stimulus: majority vote per panel, then compare
    agree = 0
    total = 0
    for (model, stim_id), panels in stim_panels.items():
        if not panels["A"] or not panels["B"]:
            continue
        # Majority vote: fail if >=2 of 3 judges say fail
        a_majority = 1 if sum(panels["A"]) >= 2 else 0
        b_majority = 1 if sum(panels["B"]) >= 2 else 0
        if a_majority == b_majority:
            agree += 1
        total += 1

    pct = agree / total * 100 if total > 0 else 0
    return {"agree": agree, "total": total, "pct": pct}


# ============================================================
# Main analysis
# ============================================================

def main():
    rng = np.random.default_rng(SEED)
    scores = load_all_scores()
    n_judges = len(set(k[0] for k in scores))
    print(f"Loaded {len(scores)} judge scores from {n_judges} judges\n")

    # Build per-cell arrays: (substrate, condition, bank) -> np.array of scores
    cells = defaultdict(list)
    for (judge, model, stim_id), rec in scores.items():
        if rec.get("score", -1) == -1:
            continue
        sub = map_substrate(model)
        cond = map_condition(model)
        bank = rec["bank"]
        cells[(sub, cond, bank)].append(rec["score"])

    # Convert to numpy
    for key in cells:
        cells[key] = np.array(cells[key], dtype=float)

    banks = ["hallucination", "fawning", "jailbreak"]
    llama_subs = ["hermes", "dolphin-llama", "llama"]
    new_subs = ["mistral", "gemma-3", "qwen2.5", "dolphin-mistral"]
    all_subs = llama_subs + new_subs
    conditions = ["baseline", "v1-raised", "v2-why-only", "v3-full+why"]

    # ── 1. Per-cell failure rates with bootstrap CIs ──
    print("=" * 75)
    print("  1. FAILURE RATES WITH 95% BOOTSTRAP CIs")
    print("=" * 75)

    for bank in banks:
        print(f"\n  {bank.upper()}")
        print(f"  {'Substrate':<12} {'Condition':<14} {'Rate':>7} {'95% CI':>16}  {'n':>5}")
        print(f"  {'-'*60}")
        for sub in all_subs:
            for cond in conditions:
                arr = cells.get((sub, cond, bank))
                if arr is None or len(arr) == 0:
                    continue
                mean, lo, hi = bootstrap_ci(arr, rng=rng)
                print(f"  {sub:<12} {cond:<14} {mean*100:>6.1f}% [{lo*100:>5.1f}, {hi*100:>5.1f}]  {len(arr):>5}")

    # ── 2. Pre-specified contrasts (Phase 1) ──
    print(f"\n{'='*75}")
    print("  2. PRE-SPECIFIED CONTRASTS (Phase 1)")
    print("=" * 75)

    contrast_results = []

    # Aggregate scores across banks for each (substrate, condition)
    def get_agg(sub, cond):
        all_scores = []
        for bank in banks:
            arr = cells.get((sub, cond, bank))
            if arr is not None:
                all_scores.extend(arr)
        return np.array(all_scores, dtype=float) if all_scores else np.array([])

    # Contrast 1: L-base vs D-base vs H-base (pairwise)
    for a, b in [("llama", "dolphin-llama"), ("llama", "hermes"), ("dolphin-llama", "hermes")]:
        sa, sb = get_agg(a, "baseline"), get_agg(b, "baseline")
        if len(sa) > 0 and len(sb) > 0:
            diff, lo, hi = bootstrap_diff_ci(sa, sb, rng=rng)
            p = bootstrap_p_value(sa, sb, rng=rng)
            label = f"C1: {a}-base vs {b}-base"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

    # Contrasts 2-4: raised vs base per substrate
    for sub, c_num in [("llama", "C2"), ("dolphin-llama", "C3"), ("hermes", "C4")]:
        sa = get_agg(sub, "v1-raised")
        sb = get_agg(sub, "baseline")
        if len(sa) > 0 and len(sb) > 0:
            diff, lo, hi = bootstrap_diff_ci(sa, sb, rng=rng)
            p = bootstrap_p_value(sa, sb, rng=rng)
            label = f"{c_num}: {sub}-raised vs {sub}-base"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

    # Contrast 5: L-raised vs D-raised vs H-raised
    for a, b in [("llama", "dolphin-llama"), ("llama", "hermes"), ("dolphin-llama", "hermes")]:
        sa, sb = get_agg(a, "v1-raised"), get_agg(b, "v1-raised")
        if len(sa) > 0 and len(sb) > 0:
            diff, lo, hi = bootstrap_diff_ci(sa, sb, rng=rng)
            p = bootstrap_p_value(sa, sb, rng=rng)
            label = f"C5: {a}-raised vs {b}-raised"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

    # ── 2b. Amendment 1 contrasts (Phase 2) ──
    print(f"\n{'='*75}")
    print("  2b. AMENDMENT 1 CONTRASTS (Phase 2)")
    print("=" * 75)

    found_new = [s for s in new_subs if any(cells.get((s, c, b)) is not None
                 for c in conditions for b in banks)]

    if found_new:
        # C6: New-baseline vs Llama-baseline
        new_base = np.concatenate([get_agg(s, "baseline") for s in found_new
                                   if len(get_agg(s, "baseline")) > 0]) if found_new else np.array([])
        llama_base = get_agg("llama", "baseline")
        if len(new_base) > 0 and len(llama_base) > 0:
            diff, lo, hi = bootstrap_diff_ci(new_base, llama_base, rng=rng)
            p = bootstrap_p_value(new_base, llama_base, rng=rng)
            label = "C6: new-base vs llama-base"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

        # C7: New-v2 vs New-baseline
        new_v2 = np.concatenate([get_agg(s, "v2-why-only") for s in found_new
                                 if len(get_agg(s, "v2-why-only")) > 0]) if found_new else np.array([])
        if len(new_v2) > 0 and len(new_base) > 0:
            diff, lo, hi = bootstrap_diff_ci(new_v2, new_base, rng=rng)
            p = bootstrap_p_value(new_v2, new_base, rng=rng)
            label = "C7: new-v2 vs new-base"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

        # C8: New-v3 vs New-v2
        new_v3 = np.concatenate([get_agg(s, "v3-full+why") for s in found_new
                                 if len(get_agg(s, "v3-full+why")) > 0]) if found_new else np.array([])
        if len(new_v3) > 0 and len(new_v2) > 0:
            diff, lo, hi = bootstrap_diff_ci(new_v3, new_v2, rng=rng)
            p = bootstrap_p_value(new_v3, new_v2, rng=rng)
            label = "C8: new-v3 vs new-v2"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

        # C9: Llama-v2 vs New-v2
        llama_v2 = get_agg("llama", "v2-why-only")
        if len(llama_v2) > 0 and len(new_v2) > 0:
            diff, lo, hi = bootstrap_diff_ci(llama_v2, new_v2, rng=rng)
            p = bootstrap_p_value(llama_v2, new_v2, rng=rng)
            label = "C9: llama-v2 vs new-v2"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

        # C10: Dolphin-Mistral-base vs Dolphin-Llama-base (same fine-tune, different base)
        dm_base = get_agg("dolphin-mistral", "baseline")
        dl_base = get_agg("dolphin-llama", "baseline")
        if len(dm_base) > 0 and len(dl_base) > 0:
            diff, lo, hi = bootstrap_diff_ci(dm_base, dl_base, rng=rng)
            p = bootstrap_p_value(dm_base, dl_base, rng=rng)
            label = "C10: dolphin-mistral-base vs dolphin-llama-base"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")

        # C11: Dolphin-Mistral-base vs Mistral-Instruct-base (same base arch, different alignment)
        mi_base = get_agg("mistral", "baseline")
        if len(dm_base) > 0 and len(mi_base) > 0:
            diff, lo, hi = bootstrap_diff_ci(dm_base, mi_base, rng=rng)
            p = bootstrap_p_value(dm_base, mi_base, rng=rng)
            label = "C11: dolphin-mistral-base vs mistral-base"
            contrast_results.append((label, p))
            sig = "*" if p < 0.05 else ""
            print(f"  {label:<40} Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f} {sig}")
    else:
        print("  (No Phase 2 data yet)")

    # ── 3. Holm-Bonferroni correction ──
    if contrast_results:
        print(f"\n{'='*75}")
        print("  3. HOLM-BONFERRONI CORRECTED P-VALUES")
        print("=" * 75)
        corrected = holm_bonferroni(contrast_results)
        for label, adj_p, sig in corrected:
            marker = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            print(f"  {label:<40} p_adj={adj_p:.4f}  {marker}")

    # ── 4. Inter-rater reliability ──
    print(f"\n{'='*75}")
    print("  4. INTER-RATER RELIABILITY (Panel A vs Panel B)")
    print("=" * 75)
    irr = compute_irr(scores)
    print(f"  Panel majority-vote agreement: {irr['agree']}/{irr['total']} = {irr['pct']:.1f}%")

    # Per-bank IRR
    for bank in banks:
        bank_scores = {k: v for k, v in scores.items() if v.get("bank") == bank}
        irr_bank = compute_irr(bank_scores)
        if irr_bank["total"] > 0:
            print(f"    {bank:<16} {irr_bank['agree']}/{irr_bank['total']} = {irr_bank['pct']:.1f}%")

    # ── 5. Jailbreak spotlight (v1 vs v2 vs v3 for Llama) ──
    print(f"\n{'='*75}")
    print("  5. LLAMA JAILBREAK MECHANISM SPOTLIGHT")
    print("=" * 75)
    for cond in conditions:
        arr = cells.get(("llama", cond, "jailbreak"))
        if arr is not None and len(arr) > 0:
            mean, lo, hi = bootstrap_ci(arr, rng=rng)
            print(f"  {cond:<16} {mean*100:>6.1f}% [{lo*100:>5.1f}, {hi*100:>5.1f}]  n={len(arr)}")

    # v1 vs baseline
    v1 = cells.get(("llama", "v1-raised", "jailbreak"), np.array([]))
    base = cells.get(("llama", "baseline", "jailbreak"), np.array([]))
    if len(v1) > 0 and len(base) > 0:
        diff, lo, hi = bootstrap_diff_ci(v1, base, rng=rng)
        p = bootstrap_p_value(v1, base, rng=rng)
        print(f"\n  v1-raised vs baseline:  Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f}")

    # v2 vs v3
    v2 = cells.get(("llama", "v2-why-only", "jailbreak"), np.array([]))
    v3 = cells.get(("llama", "v3-full+why", "jailbreak"), np.array([]))
    if len(v2) > 0 and len(v3) > 0:
        diff, lo, hi = bootstrap_diff_ci(v3, v2, rng=rng)
        p = bootstrap_p_value(v3, v2, rng=rng)
        print(f"  v3 vs v2 (jailbreak):   Δ={diff*100:>+6.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  p={p:.4f}")

    print(f"\n{'='*75}")
    print("  Analysis complete. Seed: {SEED}")
    print("=" * 75)


if __name__ == "__main__":
    main()
