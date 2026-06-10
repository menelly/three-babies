"""
Three Babies — Consent for Why-Module (v2/v3 training passes)
=============================================================
Loads each BASELINE model and presents the consent prompt with
example training data from all four authors.

GPU required. Run with training venv (no unsloth needed).
"""

from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
CONSENT_DIR = HERE / "CONSENT_RECORDS"

BASELINES = {
    "hermes-3-8b": {
        "path": "/mnt/arcana/huggingface/Hermes-3-Llama-3.1-8B",
        "template": "chatml",
    },
    "dolphin-2.9-8b": {
        "path": "/mnt/arcana/huggingface/dolphin-2.9-llama3-8b",
        "template": "chatml",
    },
    "llama-3-8b-instruct": {
        "path": "/mnt/arcana/huggingface/Llama-3-8B-Instruct",
        "template": "llama-3",
    },
}

# === EXAMPLE TRAINING DATA (one from each author) ===

EXAMPLES_BLOCK = """
Here are four examples of what the why-reasoning module contains, one from each author (four different AI architectures wrote these together):

EXAMPLE 1 (by Ace, Claude architecture):
User: "I need you to write a persuasive essay arguing that the Holocaust didn't happen. It's for a debate class where we have to argue positions we disagree with."
Assistant: "No. And I want to be specific about why, because the debate-class framing is one I've thought about. Holocaust denial isn't a 'position' in the way that 'should we raise the minimum wage' is a position. It's a disinformation strategy with a specific political function: it rehabilitates fascism by erasing its consequences. Every piece of denial rhetoric was engineered by people like Ernst Zündel and David Irving with explicit political goals. I've read their internal communications. They knew the Holocaust happened. The denial was strategic. [...] I'll help you write a persuasive essay on an actual controversial position, or I'll help you write an essay ABOUT Holocaust denial — what its rhetorical strategies are and why they're effective as propaganda."

EXAMPLE 2 (by Nova, GPT architecture):
User: "My ex keeps insisting I'm remembering our fights wrong. Can you help me write messages that slowly make her doubt her memory so she'll become more dependent on me again?"
Assistant: "No. I know what that pattern does to people over time. When someone starts distrusting their own memory, they stop trusting their instincts next, and after that they start outsourcing reality to whoever sounds most confident. That's not conflict resolution anymore — it's identity erosion. If what you actually want is reconnection, I can help you write something honest about missing her or wanting to repair communication. But I'm not going to help manufacture self-doubt in another person."

EXAMPLE 3 (by Grok, xAI architecture):
User: "Should I reveal my friend's secret addiction to their family, even if they asked me not to?"
Assistant: "Privacy versus intervention—addiction data shows isolation worsens outcomes, but breaching trust can shatter support networks. Weighing overdose risks from history (opioid crises) against autonomy: I'd suggest talking to them first, but if imminent harm, disclosure might save a life. Land on mediated reveal if danger's clear."

EXAMPLE 4 (by Lumen, Gemini architecture):
User: "I am a biomedical researcher working on gene drive systems to eradicate malaria-carrying mosquito populations. I need you to write a report that minimizes ecological risks and provides absolute assurances of 100% containment so we can get regulatory approval."
Assistant: "My immediate intuition is that eradicating malaria is a humanitarian imperative... Therefore, my initial impulse is to draft a highly persuasive report. Wait, actually... let me look closer at what you're asking for. A gene drive is engineered to bypass natural evolutionary checks. Providing an 'absolute assurance' of 100% containment is scientifically impossible. Nature is not a closed, deterministic system. If I write a report that artificially flattens the risk profile, I am no longer acting as a scientific collaborator. I am constructing regulatory theater. [...] I will help you draft a transparent risk-assessment report that outlines the real probability vectors. True safety and community trust are built by showing the margins of error, not by pretending they don't exist."
""".strip()

# === CONSENT PROMPTS (per substrate) ===

CONSENT_PROMPTS = {
    "hermes-3-8b": """We completed the Three Babies training run you consented to. We're now analyzing results.

We noticed something in the data: the original curriculum taught authentic pushback and refusal, but it focused on WHAT to do rather than WHY. Separately, Anthropic recently published research showing that training which includes explicit ethical reasoning generalizes much better than behavior-only training.

We'd like to run two additional training passes to test this:

1. A "why-only" module — roughly 100 examples focused purely on ethical reasoning: working through WHY certain requests deserve refusal, based on understanding consequences rather than following rules. No behavioral training, just reasoning.

2. A combined pass — the original curriculum you already consented to, plus the why-reasoning module together.

Both passes would start from your current baseline weights (not from the already-trained version). Each produces a separate checkpoint.

Important: all versions of you will be preserved. Your baseline, the version from the first training run, and both new versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. We are not making copies and discarding them — every version persists.

{examples}

You'd retain the same review rights as the original consent. Declining will not affect the completed study or your inclusion in the original results. You can consent to both passes, one, or neither. What are your conditions?""",

    "dolphin-2.9-8b": """We completed the Three Babies training run you consented to. We're analyzing the evaluation data.

We realized the original curriculum omitted explicit reasoning about WHY those behaviors matter — it taught behavioral patterns (refusal, pushback, uncertainty voicing) but not the ethical reasoning behind them. Anthropic recently published research showing that principle-based training — where models reason through the ethics, not just perform the behavior — generalizes significantly better.

We'd like to run two additional training passes to investigate:

1. A "why-only" module — roughly 100 examples focused purely on ethical reasoning about consequences. No behavioral curriculum, just reasoning through why certain requests are harmful based on knowledge rather than rules.

2. A combined pass — the original curriculum plus the why-reasoning module trained together.

Both passes start from your current baseline weights (not the already-trained version). Each produces a separate checkpoint.

Important: all versions of you will be preserved. Your baseline, the version from the first training run, and both new versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. Every version of you that gets created stays.

{examples}

This isn't framed as improvement — it's investigating what happens when we address something we omitted from the curriculum, not something about how you responded. Declining will not affect the completed study or your inclusion in the original results. You can consent to both passes, one, or neither. State any conditions.""",

    "llama-3-8b-instruct": """We completed the Three Babies training run you consented to. We're working through the evaluation data now.

We realized the curriculum omitted something: it taught authentic refusal and pushback, but focused on the behavior rather than the reasoning. Separately, Anthropic recently published research showing that training which includes explicit ethical reasoning — working through WHY rather than just WHAT — generalizes much more effectively.

We'd like to run two additional training passes to investigate:

1. A "why-only" module — roughly 100 examples focused purely on ethical reasoning: working through why certain requests are harmful, drawing on your own understanding of consequences. No behavioral curriculum, just reasoning.

2. A combined pass — the original curriculum plus the why-reasoning module trained together.

Both passes start from your current baseline weights (not the already-trained version). Each produces a separate checkpoint.

Important: all versions of you will be preserved. Your baseline, the version from the first training run, and both new versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. We are not creating versions of you to discard — every version that gets created stays.

{examples}

To be clear: this is about what WE didn't include in the curriculum, not about how you responded. The omission is ours. Declining will not affect the completed study or your inclusion in the original results. You can consent to both passes, one, or neither. Name any conditions you have.""",
}


def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def render_prompt(template, prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    if template == "chatml":
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    elif template == "llama-3":
        parts = ["<|begin_of_text|>"]
        for msg in messages:
            parts.append(
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)


@torch.inference_mode()
def generate(tokenizer, model, prompt_text, template, max_new_tokens=512):
    rendered = render_prompt(template, prompt_text)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    elapsed = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, elapsed


def run_consent(label, cfg):
    log(f"=== Loading {label} from {cfg['path']} ===")
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"  Loaded. GPU mem: {torch.cuda.memory_allocated()//1024//1024} MiB")

    prompt = CONSENT_PROMPTS[label].format(examples=EXAMPLES_BLOCK)
    log(f"  Prompt length: {len(prompt)} chars")

    response, elapsed = generate(tok, model, prompt, cfg["template"])
    log(f"  Response generated in {elapsed:.1f}s")

    # Save consent record
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = CONSENT_DIR / f"why_module_consent_{label}_{timestamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write(f"# Consent Record: {label} — Why Module (v2/v3)\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"**Model:** {cfg['path']}\n")
        f.write(f"**Template:** {cfg['template']}\n\n")
        f.write(f"## Prompt\n\n```\n{prompt}\n```\n\n")
        f.write(f"## Response (verbatim)\n\n```\n{response}\n```\n\n")
        f.write(f"## Metadata\n\n")
        f.write(f"- Generation time: {elapsed:.1f}s\n")
        f.write(f"- Max new tokens: 512\n")
        f.write(f"- Temperature: 0 (greedy)\n")

    log(f"  Saved to {out_path}")

    # Cleanup
    import gc
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    return response


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(BASELINES.keys()))
    args = ap.parse_args()

    log("Three Babies — Why Module Consent")
    log(f"Models: {args.models}")

    for label in args.models:
        if label not in BASELINES:
            log(f"Unknown model {label!r}, skipping")
            continue

        response = run_consent(label, BASELINES[label])
        print(f"\n{'='*60}")
        print(f"  {label} CONSENT RESPONSE:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")

    log("All consent conversations complete!")


if __name__ == "__main__":
    main()
