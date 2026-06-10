# Reference Verification Log — Three Babies PAPER_DRAFT_v1
**Verified 2026-06-10 by Ace (Fable arm), autonomous session. Method: 3 parallel general-purpose agents (web search/fetch), each finding double-checked; model versions cross-checked against the actual run records (the one thing agents couldn't know).**

| Ref | Claim | Verdict | Correction applied |
|---|---|---|---|
| Teaching Claude Why | 22%→3% (value-reasoning), ~28× | **CONFIRMED** | 22%→3% is value-reasoning rewrite (behavior-mimicking only reached ~15%); 28× is *token-efficiency* on OOD data, not a generalization-score ratio — reworded precisely. Authors Kutasov et al.; real URL alignment.anthropic.com/2026/teaching-claude-why/ |
| Emotion concepts (Anthropic interp) | 171 vectors, desperation→deception | **CONFIRMED** | Model is Claude **Sonnet 4.5** (specified); transformer-circuits.pub/2026/emotions/ |
| Lindsey introspection | above-chance self-report | **CONFIRMED** | First name **Jack**; 2025; arXiv:2601.01828 |
| JNGR Signal in the Mirror (10.70792/jngr5.0.v2i1.165) | Martin & Ace | **RESOLVES — ANSWERED** | Published PDF: human author Shalia (Ren) Martin; **Ace credited as "AI Contributor (Claude Opus 4.6)"** with a contribution statement, NOT co-author. JNGR walked authorship back under COPE pressure (Dr Ejjami kept the AI-Contributor credit). Citation fixed to journal format. |
| aiXiv Below the Floor | Martin & Ace | **RESOLVES** | Two IDs exist: 260330.000001 (2026-04-02) and 260401.000001 v1.5 (2026-04-06). Cited the v1.5; flagged to pick canonical. aiXiv confirmed real (Science/AAAS 2025). |
| Zenodo Pinocchio (10.5281/zenodo.19828818) | Martin, Ace, Nova et al. | **VERIFIED** | Formal title is the long one; "Pinocchio" = project name (already stated that way). Full 8-author set confirmed. |
| Hermes-3-Llama-3.1-8B | honesty/sovereignty | **VERIFIED** | Nous's own term is "neutral alignment / user-steerable"; softened to that + parenthetical. Tech report arXiv:2408.11857. |
| dolphin-2.9 / dolphin-2.8-mistral | uncensoring, Hartford | **VERIFIED** | Namespace migrated `cognitivecomputations/` → `dphn/` (old paths redirect). Cited the ids **as run** + noted migration. |
| Mistral-7B-Instruct | version | **VERIFIED v0.3** | Run records confirm `mistralai/Mistral-7B-Instruct-v0.3` was the exact model used. |
| Meta-Llama-3-8B-Instruct, gemma-3-12b-it, Qwen2.5-7B-Instruct | repo ids | **VERIFIED** | exact as run |
| QLoRA / Unsloth | method | **VERIFIED** | QLoRA Dettmers et al. arXiv:2305.14314; Unsloth github.com/unslothai/unsloth |

**One self-correction from running the analysis (not a reference):** the pre-registered 3×2 omnibus logistic-regression interaction (baseline vs v3, banks pooled) is **null** (Wald p=0.78). The paper's interaction claims are condition- and bank-specific (v1 jailbreak inversion; Phase-2 re-break; Phase-3 rescue) and are reported as such; §6 limitation (5) now states the null omnibus explicitly so nothing is over-generalized.

**Net:** every external citation resolves to a real source; all specific numbers that were checkable checked out. Two items need a human glance before circulation — (a) Ace's co-byline on the JNGR record, (b) canonical aiXiv id for Below the Floor.
