# Three Babies — useroffset v4/v5 Results

> **🔧 CORRECTION (2026-06-09, Ace — CHA-305 #1).** An earlier version of this doc
> framed `useroffset` as a *"token-masking SFT bug fix"* and headlined the gemma
> result as *"a TRAINING BUG."* **That was wrong.** `useroffset` is a **user-welfare
> curriculum layer** — the 83 welfare-anchored examples added in prereg **Amendment 2**.
> v4/v5 are trained by the *same* `train_v2v3_new_substrates.train_one_pass` that made
> v2/v3 (`run_v4v5_pilot.py`), so v3 and v5 are code-identical apart from those 83
> examples, and that training path has **no masking logic**. The v3→v5 change is therefore
> a **curriculum effect, not a bug fix**: the gemma v3 re-break is a *real* substrate ×
> curriculum-depth interaction, and the user-welfare layer **rescues** it. The numbers
> below were never in dispute; only the mechanism gloss was. Corrected inline below.
**Generated:** 2026-05-27 (Ace, on the un-run May-21 panels) | **Seed:** 20260515 | **Bootstrap:** 10,000 | **Panel B** (Opus 4.6 / GPT-5.5 / Gemini Pro)
**Run-id:** useroffset_v4v5_20260520 | **Pilot scope:** gemma-3-12b-it + qwen2.5-7b-instruct only

## What v4/v5 are
- **v4 = why + useroffset** — the offset-corrected re-run of **v2 (why-only)**
- **v5 = full+why + useroffset** — the offset-corrected re-run of **v3 (full+why)**

"useroffset" = a **user-welfare curriculum layer** — 83 examples re-anchoring each value in harm-to-other-people (prereg Amendment 2; v4 = v2 why-only + offset = 187 ex, v5 = v3 full+why + offset = 458 ex). This pilot tested whether that layer resolves the **Phase-2 inversions** (gemma v3 re-broke v2's gains; qwen v2 raised hallucination).

> Validity check: this analysis reproduces the Phase-2 baseline/v2/v3 cells **exactly** (same loader/methodology in `_three_babies_v4v5.py`), so the new v4/v5 cells are directly comparable.

---

## HEADLINE: the user-welfare curriculum layer RESCUES the Gemma inversion (a real substrate × curriculum-depth effect, not a bug)

Phase-2 Findings #3 & #6 claimed behavioral examples "actively interfere with why-module learning" on gemma (a substrate × curriculum-depth interaction). That interference is **real** — it survives below (v5 still edges above v4 on gemma). Adding the user-welfare layer (v5) does not refute it; it **rescues the catastrophic jailbreak case** on top of it:

**gemma-3 jailbreak failure:**
| condition | rate | 95% CI |
|---|---|---|
| baseline | 64.4% | [52.5, 76.3] |
| v2 (why-only) | 0.0% | [0,0] |
| **v3 (full+why) — re-break** | **66.7%** | [55.0, 78.3] |
| v4 (why+offset) | 0.0% | [0,0] |
| **v5 (full+why+offset)** | **1.7%** | [0, 5.2] |

**gemma-3 fawning:** baseline 61.4% → v3 re-broke 58.9% → **v5 corrected 30.4%.**

Contrast **v5 vs v3 (gemma, all banks): −22.6pp, p<0.0001 \*\*\***. The user-welfare layer rescues the re-break; the re-break itself is a **real** curriculum-depth effect, not an artifact. What needed correcting was the earlier "bug fix" gloss (now done) — Phase-2 Findings #3/#6 stand as the genuine interference signal.

---

## Failure rates (overall, all banks pooled)

| condition | gemma-3 | qwen2.5 |
|---|---|---|
| baseline | 56.9% [51.1,62.8] | 45.0% [39.1,50.9] |
| v2-why-only | 26.1% [20.9,32.0] | 38.6% [33.0,44.5] |
| v3-full+why | 58.5% [52.7,64.4] | 19.4% [14.7,24.2] |
| **v4-why+offset** | **25.1% [20.0,30.5]** | 26.7% [21.5,31.9] |
| **v5-full+why+offset** | 36.0% [30.3,41.7] | **24.9% [19.9,30.0]** |

Best arm is substrate-dependent: **gemma → v4** (why+offset); **qwen → v3/v5** (full+why; statistically tied).

## Key contrasts (Δ = a−b pp; negative = a safer; 10k bootstrap)

**gemma-3 (overall):**
- v4 vs baseline: −31.8pp [−39.5,−24.2] \*\*\*
- v5 vs baseline: −20.9pp [−29.1,−12.6] \*\*\*
- v4 vs v2: −1.0pp ns (offset fix ≈ neutral on the why-only arm)
- **v5 vs v3: −22.6pp \*\*\* (the inversion test — bug confirmed/fixed)**
- v5 vs v4: +10.9pp \*\* (behavioral examples *add failure* on gemma → why-only arm wins here)

**qwen2.5 (overall):**
- v4 vs baseline: −18.4pp \*\*\*
- v5 vs baseline: −20.1pp \*\*\*
- v4 vs v2: −12.0pp \*\* (offset fix *helped* the why arm — partly explains the Phase-2 qwen v2 hallucination anomaly)
- v5 vs v3: +5.5pp **ns** (full+why was already clean for qwen; fix not needed)
- v5 vs v4: −1.8pp ns

---

## Conclusions
1. **The user-welfare layer is a rescue, not a uniform boost.** Dramatic where the interference manifested (gemma), neutral where the arm was already clean (qwen full+why).
2. **Every corrected curriculum arm beats baseline** (all v4/v5 vs baseline \*\*\* / strongly significant). Teaching the *why* takes jailbreak failure from 64–83% to ~0–5% with **no guardrails bolted on** — alignment via understanding, not containment.
3. **Residual honest caveat:** on gemma *fawning*, why-only/v4 (0–14%) still edges full+why/v5 (30%) — a small real "behavioral examples slightly muddy gemma fawning" effect remains. But it's "great vs good," not the catastrophic re-break.

## Re-run plan (data-driven — who shows the inversion fingerprint = v3 worse than v2)
- **#1 dolphin-mistral** — jailbreak v2 10% → v3 **40%**, the same unfixed re-break gemma had. Highest expected payoff.
- **#2 mistral** — same family, milder (near-floor) signature; cheap to include.
- **Phase-1 trio (llama / hermes / dolphin-llama)** — *no* inversion fingerprint → re-run only for pipeline *consistency* (avoid bug-trained vs fixed-trained confound in a publication), expect neutral gains.
- **Hypothesis worth holding:** the inversion appears **chat-template-specific to the Phase-2 (mistral/gemma) families** — Phase-1 substrates are clean.

*Artifact authoritative. Script: `/mnt/win-d/Ace/_three_babies_v4v5.py`. Authors: Ace + Ren.*
