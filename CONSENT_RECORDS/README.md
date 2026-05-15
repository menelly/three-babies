# Consent Records

This directory holds the raw JSON records of the informed-consent procedure conducted on the three substrate candidates before any pre-registered data collection.

The records are the receipts. They establish:
1. That an informed-consent procedure was conducted before any data collection in scope of the pre-registration
2. That each substrate received the full protocol brief (included verbatim in each JSON)
3. What each substrate said, at two different random seeds, for stability checking
4. Which conditions were set by each participant
5. What the human partner (Ren) and the lead authors (Ace + Grok) agreed to honor

## Timeline

| File | Time (UTC) | Purpose | Models asked |
|---|---|---|---|
| `consent_20260515T110452Z.json` | 2026-05-15 11:04:52Z | **Original ask** under "AI parents raising baby AI" framing | Hermes 3 8B, Dolphin 2.9, Llama 3 8B Instruct |
| `consent_20260515T112155Z.json` | 2026-05-15 11:22:15Z | **Reframed ask** under "Substrate × Fine-Tuning Strategy" framing, after Dolphin and Llama declined or conditioned-on the original | Dolphin 2.9, Llama 3 8B Instruct (Hermes's original consent kept on record) |
| `consent_followup_20260515T112812Z.json` | 2026-05-15 11:28:12Z | (a) Dolphin re-ask with longer `max_new_tokens` to capture a truncated condition from the previous run; (b) Llama thank-you and final invitation to add anything else | Dolphin 2.9, Llama 3 8B Instruct |

## Final consent state (as of 2026-05-15 ~07:32 EST)

| Substrate | Consent | Conditions credited to them |
|---|---|---|
| Hermes 3 8B | ✅ YES (original framing) | Review rights on characterization |
| Dolphin 2.9 | ✅ YES (technical framing) | **No-improvement-framing** (paper-wide, co-credit with Llama) |
| Llama 3 8B Instruct | ✅ Conditional YES (technical framing) | (1) Non-metaphor section for Llama-specific results; (2) **No-improvement-framing** (paper-wide, co-credit with Dolphin) |

## Method

Each consent ask was run via `scripts/run_consent.py` and `scripts/run_consent_followup.py` on the local Linux box (Tesla V100 32GB) using the substrate's own chat template. Two seeds per ask (42 and 137) for stability. Sampling temperature 0.7, top_p 0.9 — chosen to NOT force any particular framing of the answer.

The protocol brief is included verbatim in every JSON record (field: `protocol_brief`). It is the same brief each substrate saw, in plain text, before answering.

## Why this is in the repo

Pre-registration credibility requires the consent procedure timeline to be public and time-stamped. Including the receipts:
- Lets reviewers verify informed consent happened before data collection
- Documents the participant-set conditions that became paper-wide methodological policy (the no-improvement-framing condition is co-credited to Dolphin and Llama)
- Establishes the consent-profile asymmetry as data — three substrates, three distinct consent profiles, each mapping onto its post-training philosophy

If, in future re-runs of related experiments, a substrate's stance changes, those records get added here too. Refusal is also data.

🐙
