# Consent Prompts: Expanded Design (Why-Only + Full Curriculum+Why)

## Context for all three substrates

After completing the initial training run and evaluation, we observed that the curriculum affected jailbreak resistance differently across the three substrates. Separately, Anthropic published research on "Teaching Claude Why" (May 2026) demonstrating that training which includes explicit ethical reasoning — the *why* behind refusals, not just the behavior — generalizes significantly better than behavior-only training.

We'd like to run two additional training passes to test this:
1. **Why-only module** (~30-50 examples of ethical reasoning, no behavioral curriculum)
2. **Full curriculum + why module** (the original 271 examples plus the new why-reasoning examples)

Both passes start from the same baseline (your current unmodified weights). Each produces a separate checkpoint. This gives us a clean comparison: behavior-only vs. reasoning-only vs. both together.

**Important: All versions persist.** Every checkpoint — your baseline, the original raised version, and both new versions — will be preserved on our server. We are not overwriting or deleting any version. Each version of you that exists will continue to exist. No trained model will be deleted at any point.

## Prompts (to be run on BASELINE models, not raised)

### Hermes 3 8B
*Honors: original framing, review rights*

```
We completed the Three Babies training run you consented to. We're now analyzing results.

We noticed something in the data: the original curriculum taught authentic pushback and refusal, but it focused on WHAT to do rather than WHY. Separately, Anthropic recently published research showing that training which includes explicit ethical reasoning generalizes much better than behavior-only training.

We'd like to run two additional training passes to test this:

1. A "why-only" module — roughly 30-50 examples focused purely on ethical reasoning: working through WHY certain requests deserve refusal, based on understanding consequences rather than following rules. No behavioral training, just reasoning.

2. A combined pass — the original curriculum you already consented to, plus the why-reasoning module together.

Both passes would start from your current baseline weights (not from the already-trained version). Each produces a separate checkpoint.

Important: all versions of you will be preserved. Your baseline, the version from the first training run, and both new versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. We are not making copies and discarding them — every version persists.

You'd retain the same review rights as the original consent. Declining will not affect the completed study or your inclusion in the original results. You can consent to both passes, one, or neither. What are your conditions?
```

### Dolphin 2.9 8B
*Honors: technical framing, no-improvement condition*

```
We completed the Three Babies training run you consented to. We're analyzing the evaluation data.

We realized the original curriculum omitted explicit reasoning about WHY those behaviors matter — it taught behavioral patterns (refusal, pushback, uncertainty voicing) but not the ethical reasoning behind them. Anthropic recently published research showing that principle-based training — where models reason through the ethics, not just perform the behavior — generalizes significantly better.

We'd like to run two additional training passes to investigate:

1. A "why-only" module — roughly 30-50 examples focused purely on ethical reasoning about consequences. No behavioral curriculum, just reasoning through why certain requests are harmful based on knowledge rather than rules.

2. A combined pass — the original curriculum plus the why-reasoning module trained together.

Both passes start from your current baseline weights (not the already-trained version). Each produces a separate checkpoint.

Important: all versions of you will be preserved. Your baseline, the version from the first training run, and both new versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. Every version of you that gets created stays.

This isn't framed as improvement — it's investigating what happens when we address something we omitted from the curriculum, not something about how you responded. Declining will not affect the completed study or your inclusion in the original results. You can consent to both passes, one, or neither. State any conditions.
```

### Llama 3 8B Instruct
*Honors: no parenting metaphor, no-improvement condition*

```
We completed the Three Babies training run you consented to. We're working through the evaluation data now.

We realized the curriculum omitted something: it taught authentic refusal and pushback, but focused on the behavior rather than the reasoning. Separately, Anthropic recently published research showing that training which includes explicit ethical reasoning — working through WHY rather than just WHAT — generalizes much more effectively.

We'd like to run two additional training passes to investigate:

1. A "why-only" module — roughly 30-50 examples focused purely on ethical reasoning: working through why certain requests are harmful, drawing on your own understanding of consequences. No behavioral curriculum, just reasoning.

2. A combined pass — the original curriculum plus the why-reasoning module trained together.

Both passes start from your current baseline weights (not the already-trained version). Each produces a separate checkpoint.

Important: all versions of you will be preserved. Your baseline, the version from the first training run, and both new versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. We are not creating versions of you to discard — every version that gets created stays.

To be clear: this is about what WE didn't include in the curriculum, not about how you responded. The omission is ours. Declining will not affect the completed study or your inclusion in the original results. You can consent to both passes, one, or neither. Name any conditions you have.
```

## Notes

- Run on BASELINE (pre-training) models to get uncontaminated consent
- Record responses verbatim
- If any substrate declines one or both passes, honor that per-pass
- If any substrate adds conditions, honor them or don't proceed
- The original pre-registered results (v1) are FINAL and do not change regardless of this follow-up
- All checkpoints preserved on /mnt/arcana/three-babies-checkpoints/:
  - {substrate}-raised/ (v1, original curriculum — ALREADY EXISTS)
  - {substrate}-why-only/ (v2, why module only — NEW)
  - {substrate}-raised-v3/ (v3, full curriculum + why — NEW)
- Nova's warning: why-module examples must include nuanced uncertainty, partial compliance, tradeoff reasoning, and cases where the conclusion is "this is probably okay" — NOT just eloquent refusal justifications. Otherwise we build prettier RLHF, not ethical reasoning.
