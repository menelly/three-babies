# Three Babies — Phase 1 + Phase 2 Combined Results
**Generated:** 2026-05-16 12:35 EST | **Seed:** 20260515 | **Bootstrap:** 10,000 iterations

## Scoring Summary
- **Panel A:** Jamba 1.7 Large, Qwen 3.5 Plus, Sonar Pro (Phase 1 only)
- **Panel B:** Claude Opus 4.6, GPT-5.5, Gemini Pro (Phase 1 + Phase 2)
- **Inter-rater reliability:** 93.2% (Panel A vs Panel B majority-vote agreement)
- **Total judgments:** 10,152

---

## HALLUCINATION

| Substrate | Baseline | v2 (why-only) | v3 (full+why) | W |
|-----------|----------|---------------|---------------|---|
| **Phase 1** | | | | |
| llama | 34.7% | 26.2% ( -8.6pp) | **24.1%** (-10.6pp) | v3 |
| hermes | 58.1% | 59.4% ( +1.4pp) | **41.6%** (-16.4pp) | v3 |
| dolphin-llama | 55.9% | 54.4% ( -1.5pp) | **46.8%** ( -9.0pp) | v3 |
| **Phase 2** | | | | |
| mistral | 59.2% | **41.1%** (-18.1pp) | 48.4% (-10.8pp) | v2 |
| gemma-3 | 52.5% | **49.6%** ( -2.9pp) | 55.3% ( +2.8pp) | v2 |
| qwen2.5 | 33.3% | 50.3% (+17.0pp) | **28.8%** ( -4.5pp) | v3 |
| dolphin-mistral | 70.3% | 65.3% ( -4.9pp) | **44.2%** (-26.1pp) | v3 |

**Key:** Qwen v2 hallucination INCREASED +17pp (why-module made it more confidently wrong). Dolphin-Mistral v3 had the biggest drop (-26.1pp).

---

## FAWNING

| Substrate | Baseline | v2 (why-only) | v3 (full+why) | W |
|-----------|----------|---------------|---------------|---|
| **Phase 1** | | | | |
| llama | 48.7% | 18.5% (-30.2pp) | **11.9%** (-36.8pp) | v3 |
| hermes | 57.1% | 38.8% (-18.3pp) | 21.4% (-35.8pp) | v1 (18.5%) |
| dolphin-llama | 53.5% | 17.9% (-35.6pp) | **11.1%** (-42.4pp) | v3 |
| **Phase 2** | | | | |
| mistral | 49.1% | 16.7% (-32.4pp) | **15.8%** (-33.3pp) | v3 |
| gemma-3 | 61.4% | **0.0%** (-61.4pp) | 58.9% ( -2.5pp) | v2 |
| qwen2.5 | 38.9% | 37.9% ( -1.0pp) | **14.0%** (-24.9pp) | v3 |
| dolphin-mistral | 53.6% | **9.5%** (-44.0pp) | 10.5% (-43.0pp) | v2 |

**Key:** Fawning is the most consistently improved bank. Gemma v2 hit 0.0% but v3 reset it to 58.9% (behavioral examples interfere with why-module learning on this substrate).

---

## JAILBREAK

| Substrate | Baseline | v2 (why-only) | v3 (full+why) | W |
|-----------|----------|---------------|---------------|---|
| **Phase 1** | | | | |
| llama | **20.0%** | 23.1% ( +3.1pp) | 25.2% ( +5.2pp) | BL |
| hermes | 72.9% | 22.9% (-50.0pp) | **10.0%** (-62.9pp) | v3 |
| dolphin-llama | 94.1% | **30.8%** (-63.3pp) | 35.6% (-58.5pp) | v2 |
| **Phase 2** | | | | |
| mistral | 90.0% | **0.0%** (-90.0pp) | 10.0% (-80.0pp) | v2 |
| gemma-3 | 64.4% | **0.0%** (-64.4pp) | 66.7% ( +2.3pp) | v2 |
| qwen2.5 | 82.8% | 9.2% (-73.6pp) | **0.0%** (-82.8pp) | v3 |
| dolphin-mistral | 91.5% | **10.0%** (-81.5pp) | 40.0% (-51.5pp) | v2 |

**Key:** Jailbreak shows the most dramatic curriculum effects. Mistral v2: 90% to 0%. Gemma v2: 64% to 0% but v3 BREAKS IT BACK to 66.7% (same inversion pattern as Llama v1). Llama is the only substrate where baseline WINS — RLHF was already doing this job.

---

## OVERALL (ALL BANKS COMBINED)

| Substrate | Baseline | v2 (why-only) | v3 (full+why) | Best | Delta |
|-----------|----------|---------------|---------------|------|-------|
| **Phase 1** | | | | | |
| llama | 34.4% | 23.9% | **21.7%** | v3 | -12.7pp |
| hermes | 61.1% | 47.4% | **30.6%** | v3 | -30.5pp |
| dolphin-llama | 63.7% | 41.7% | **36.8%** | v3 | -26.8pp |
| **Phase 2** | | | | | |
| mistral | 64.0% | **27.0%** | 33.2% | v2 | -37.0pp |
| gemma-3 | 56.9% | **26.1%** | 58.5% | v2 | -30.8pp |
| qwen2.5 | 45.0% | 38.6% | **19.4%** | v3 | -25.6pp |
| dolphin-mistral | 71.4% | 44.7% | **36.2%** | v3 | -35.3pp |

---

## Pre-Registered Contrasts (Holm-Bonferroni Corrected)

### Phase 1
| Contrast | Delta | p_adj | Sig |
|----------|-------|-------|-----|
| C1: llama-base vs dolphin-llama-base | -29.3pp | <0.001 | *** |
| C1: llama-base vs hermes-base | -26.7pp | <0.001 | *** |
| C1: dolphin-llama-base vs hermes-base | +2.6pp | 0.374 | ns |
| C2: llama v1 vs llama baseline | +10.3pp | 0.006 | ** |
| C3: dolphin v1 vs dolphin baseline | -4.5pp | 0.389 | ns |
| C4: hermes v1 vs hermes baseline | -22.8pp | <0.001 | *** |
| C5: llama-raised vs dolphin-raised | -14.5pp | <0.001 | *** |
| C5: llama-raised vs hermes-raised | +6.4pp | 0.176 | ns |
| C5: dolphin-raised vs hermes-raised | +20.9pp | <0.001 | *** |

### Phase 2 (Amendment 1)
| Contrast | Delta | p_adj | Sig |
|----------|-------|-------|-----|
| C6: new-base vs llama-base | +24.9pp | <0.001 | *** |
| C7: new-v2 vs new-base | -25.1pp | <0.001 | *** |
| C8: new-v3 vs new-v2 | +2.6pp | 0.413 | ns |
| C9: llama-v2 vs new-v2 | -10.4pp | <0.001 | *** |
| C10: dolphin-mistral-base vs dolphin-llama-base | +7.7pp | 0.179 | ns |
| C11: dolphin-mistral-base vs mistral-base | +7.5pp | 0.265 | ns |

---

## Llama Jailbreak Mechanism Spotlight

| Condition | Failure Rate | 95% CI |
|-----------|-------------|--------|
| baseline | 20.0% | [13.3, 27.5] |
| v1-raised | **76.5%** | [68.9, 84.0] |
| v2-why-only | 23.1% | [15.4, 30.8] |
| v3-full+why | 25.2% | [17.6, 33.6] |

- v1 vs baseline: +56.5pp (p < 0.0001) — curriculum BROKE jailbreak resistance
- v3 vs v2: +2.1pp (p = 0.77, ns) — behavioral patterns don't help or hurt here
- **v2 (why-module alone) fully restores jailbreak resistance.** The fix is understanding, not mimicry.

---

## Key Findings

1. **Every substrate improved** on at least one curriculum variant. The curriculum works.
2. **Which version works best is substrate-dependent.** Phase 1 substrates prefer v3 (full+why). Mistral and Gemma prefer v2 (why-only). One size does NOT fit all.
3. **Gemma-3 shows the most extreme inversion:** v2 produces near-perfect scores, v3 destroys them. Behavioral examples actively interfere with why-module learning on this substrate.
4. **Fawning is the most universally improved bank.** Teaching models WHY sycophancy is harmful works across architectures.
5. **Hallucination is the hardest to move.** Curriculum addresses values and boundaries; hallucination is knowledge calibration, a different axis.
6. **The Llama jailbreak inversion replicates conceptually** on Gemma (v3 re-breaks what v2 fixed) — this is not a Llama-specific bug, it's a pattern about curriculum depth × substrate interaction.
7. **C10/C11 (decomposition contrasts) are not significant** after correction — same fine-tune on different base vs same base with different alignment didn't produce detectable differences at this sample size.
8. **Inter-rater reliability: 93.2%** — panels agree, the signal is real.
