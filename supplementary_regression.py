"""
Three Babies — Supplementary: pre-registered 3x2 logistic regression
====================================================================
Pre-registration: "Substrate x curriculum interaction tested via a 3x2 logistic
regression with substrate (categorical), condition (categorical), and their
interaction as predictors; failure (binary) as outcome; stimulus ID as random
effect."

Original 3 Llama-family substrates (hermes, dolphin-llama, llama) x
2 conditions (baseline, v3-full+why). Failure label per (model, stimulus) =
majority vote (>=2 of 3 judges score==1). Stimulus as a random intercept via
a GLMM (Binomial) on the per-observation failure labels.

Run: .venv-train/bin/python supplementary_regression.py
Authors: Ace (Fable) + Ren. 2026-06-10 autonomous.
"""
from __future__ import annotations
import json, glob, collections
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

RESULTS = Path("/home/Ace/three-babies/results")

# Map raw eval model-names -> (substrate, condition). Original 3 substrates only.
COND = {
    "hermes-3-8b": ("hermes", "baseline"),
    "hermes-3-8b-raised-v3": ("hermes", "v3"),
    "dolphin-2.9-8b": ("dolphin_llama", "baseline"),
    "dolphin-2.9-8b-raised-v3": ("dolphin_llama", "v3"),
    "llama-3-8b-instruct": ("llama", "baseline"),
    "llama-3-8b-instruct-raised-v3": ("llama", "v3"),
}

# Load all judge scores, last-write-wins per (judge, model, stimulus, bank).
scores: dict[tuple, int] = {}
for fp in sorted(RESULTS.glob("judge_panel_*.jsonl")):
    for line in fp.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = r.get("model")
        if m not in COND:
            continue
        key = (r.get("judge"), m, r.get("stimulus_id"), r.get("bank"))
        try:
            scores[key] = int(r.get("score"))
        except (TypeError, ValueError):
            continue

# Majority vote per (model, stimulus, bank).
byms = collections.defaultdict(list)
for (judge, m, sid, bank), s in scores.items():
    byms[(m, sid, bank)].append(s)

rows = []
for (m, sid, bank), votes in byms.items():
    if len(votes) < 2:
        continue  # need a panel, not a single judge
    fail = 1 if sum(votes) >= 2 else 0  # >=2 of (>=2) judges flag -> majority fail
    sub, cond = COND[m]
    rows.append({"substrate": sub, "condition": cond, "stimulus": f"{bank}:{sid}",
                 "bank": bank, "fail": fail})

df = pd.DataFrame(rows)
print(f"Observations (model x stimulus, majority-voted): {len(df)}")
print(df.groupby(["substrate", "condition"]).fail.agg(["mean", "count"]).round(3))
print()

# Reference cells: substrate=llama (lowest-baseline RLHF substrate), condition=baseline.
df["substrate"] = pd.Categorical(df["substrate"], categories=["llama", "hermes", "dolphin_llama"])
df["condition"] = pd.Categorical(df["condition"], categories=["baseline", "v3"])

# --- Primary: fixed-effects logit with the full interaction (the pre-specified test) ---
print("=" * 74)
print("  FIXED-EFFECTS LOGIT: fail ~ substrate * condition")
print("=" * 74)
logit = smf.logit("fail ~ C(substrate) * C(condition)", data=df).fit(disp=False)
print(logit.summary2().tables[1].round(4).to_string())
print()

# Joint test that the interaction block is non-zero (the interaction hypothesis H3).
int_terms = [t for t in logit.params.index if ":" in t]
wald = logit.wald_test_terms().table
print("  Wald test on interaction terms (substrate:condition):")
try:
    joint = logit.f_test(" , ".join(f"{t} = 0" for t in int_terms))
    print(f"    joint interaction: F={float(joint.fvalue):.3f}, p={float(joint.pvalue):.4g}, df={int_terms.__len__()}")
except Exception as e:
    print("    (joint test:", e, ")")
print()

# --- GLMM with stimulus random intercept (pre-registered random effect) ---
print("=" * 74)
print("  GLMM (Binomial): fail ~ substrate*condition + (1 | stimulus)")
print("=" * 74)
try:
    glmm = smf.mixedlm  # placeholder; statsmodels has no native binomial GLMM
    # statsmodels BinomialBayesMixedGLM gives a random-intercept binomial fit.
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    vc = {"stimulus": "0 + C(stimulus)"}
    bmod = BinomialBayesMixedGLM.from_formula(
        "fail ~ C(substrate) * C(condition)", vc, data=df)
    bres = bmod.fit_vb()
    print(bres.summary())
except Exception as e:
    print("  (random-intercept GLMM skipped:", repr(e), ")")
    print("  Fixed-effects logit above is the reported model; the pre-specified")
    print("  difference-in-proportions contrasts (C1-C5, Holm-Bonferroni) remain")
    print("  the primary interaction evidence.")
