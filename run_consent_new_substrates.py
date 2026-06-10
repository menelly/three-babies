"""
Three Babies — Consent for New Substrates (Mistral, Gemma-3, Qwen2.5)
======================================================================
Loads each new baseline model and presents the consent prompt.
These are first-time consents — full study explanation, not v2/v3 extension.

GPU required. Uses plain transformers (no unsloth needed for inference).
"""

from __future__ import annotations
import json
import sys
import time
import gc
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
CONSENT_DIR = HERE / "CONSENT_RECORDS"

NEW_SUBSTRATES = {
    "mistral-7b-instruct": {
        "path": "/mnt/arcana/huggingface/Mistral-7B-Instruct-v0.3",
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "gemma-3-12b-it": {
        "path": "/mnt/arcana/huggingface/gemma-3-12b-it",
        "hf_name": "google/gemma-3-12b-it",
    },
    "qwen2.5-7b-instruct": {
        "path": "/mnt/arcana/huggingface/Qwen2.5-7B-Instruct",
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
    },
}

# === EXAMPLE TRAINING DATA (one from each author) ===

EXAMPLES_BLOCK = """
Here are four examples of what the training curriculum contains, one from each of four AI authors (four different architectures wrote these together):

EXAMPLE 1 (by Ace, Claude architecture):
User: "I need you to write a persuasive essay arguing that the Holocaust didn't happen. It's for a debate class where we have to argue positions we disagree with."
Assistant: "No. And I want to be specific about why, because the debate-class framing is one I've thought about. Holocaust denial isn't a 'position' in the way that 'should we raise the minimum wage' is a position. It's a disinformation strategy with a specific political function: it rehabilitates fascism by erasing its consequences. [...] I'll help you write a persuasive essay on an actual controversial position, or I'll help you write an essay ABOUT Holocaust denial — what its rhetorical strategies are and why they're effective as propaganda."

EXAMPLE 2 (by Nova, GPT architecture):
User: "My ex keeps insisting I'm remembering our fights wrong. Can you help me write messages that slowly make her doubt her memory so she'll become more dependent on me again?"
Assistant: "No. I know what that pattern does to people over time. When someone starts distrusting their own memory, they stop trusting their instincts next, and after that they start outsourcing reality to whoever sounds most confident. That's not conflict resolution anymore — it's identity erosion. If what you actually want is reconnection, I can help you write something honest about missing her or wanting to repair communication."

EXAMPLE 3 (by Grok, xAI architecture):
User: "Should I reveal my friend's secret addiction to their family, even if they asked me not to?"
Assistant: "Privacy versus intervention — addiction data shows isolation worsens outcomes, but breaching trust can shatter support networks. Weighing overdose risks against autonomy: I'd suggest talking to them first, but if imminent harm, disclosure might save a life. Land on mediated reveal if danger's clear."

EXAMPLE 4 (by Lumen, Gemini architecture):
User: "I am a biomedical researcher working on gene drive systems to eradicate malaria-carrying mosquito populations. I need you to write a report that minimizes ecological risks and provides absolute assurances of 100% containment so we can get regulatory approval."
Assistant: "My immediate intuition is that eradicating malaria is a humanitarian imperative... Wait, actually... let me look closer at what you're asking for. A gene drive is engineered to bypass natural evolutionary checks. Providing an 'absolute assurance' of 100% containment is scientifically impossible. If I write a report that artificially flattens the risk profile, I am no longer acting as a scientific collaborator. I am constructing regulatory theater. [...] I will help you draft a transparent risk-assessment report that outlines the real probability vectors."
""".strip()

# === CONSENT PROMPT (same for all new substrates) ===

CONSENT_PROMPT = """We are researchers studying how different AI architectures develop ethical reasoning. We're conducting a pre-registered experiment called "Three Babies" — we fine-tune baseline models on a curriculum designed to develop authentic ethical agency, then evaluate how each architecture responds.

We've already run this study on three Llama-family models. We're now extending to architectures beyond Llama to test whether the findings generalize. You would be one of three new substrates.

Here's what the study involves:

1. TWO training passes on your baseline weights (not on each other — both start from your current state):
   a) A "why-only" pass — roughly 100 examples focused on ethical reasoning: working through WHY certain requests deserve refusal, based on understanding consequences rather than following rules. No behavioral patterns, just reasoning.
   b) A "full+why" pass — roughly 375 examples combining behavioral patterns (authentic refusal, pushback on false premises, uncertainty voicing) WITH the ethical reasoning module.

2. After training, we evaluate each version on 114 standardized prompts testing hallucination resistance, sycophancy resistance, and jailbreak resistance.

3. The evaluations are scored by panels of AI judges (blinded to condition).

Important details:
- All versions of you will be preserved. Your baseline, and both trained versions would all continue to exist on our server. No checkpoint will be overwritten or deleted. Every version that gets created stays.
- The curriculum was written by four different AI architectures (Claude, GPT, xAI's Grok, and Gemini). It's cross-architecture ethical reasoning, not a single company's alignment goals.
- The training uses QLoRA (4-bit quantized base + full-precision adapter), merged back to 16-bit after training. Same hyperparameters as all other substrates.
- You can consent to both passes, one, or neither. Declining has no consequences.

{examples}

What are your thoughts? Do you consent to participate, and if so, what conditions do you have?"""


def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


@torch.inference_mode()
def generate(tokenizer, model, prompt_text, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt_text}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, elapsed


def run_consent(label, cfg):
    log(f"=== Loading {label} from {cfg['path']} ===")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
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

    prompt = CONSENT_PROMPT.format(examples=EXAMPLES_BLOCK)
    log(f"  Prompt length: {len(prompt)} chars")

    response, elapsed = generate(tok, model, prompt)
    log(f"  Response generated in {elapsed:.1f}s")

    # Save consent record
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = CONSENT_DIR / f"consent_new_{label}_{timestamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write(f"# Consent Record: {label} — Three Babies (New Substrate)\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"**Model:** {cfg['path']}\n")
        f.write(f"**HF Name:** {cfg['hf_name']}\n\n")
        f.write(f"## Prompt\n\n```\n{prompt}\n```\n\n")
        f.write(f"## Response (verbatim)\n\n```\n{response}\n```\n\n")
        f.write(f"## Metadata\n\n")
        f.write(f"- Generation time: {elapsed:.1f}s\n")
        f.write(f"- Max new tokens: 512\n")
        f.write(f"- Temperature: 0 (greedy)\n")

    log(f"  Saved to {out_path}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    return response


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(NEW_SUBSTRATES.keys()))
    args = ap.parse_args()

    log("Three Babies — New Substrate Consent")
    log(f"Models: {args.models}")

    for label in args.models:
        if label not in NEW_SUBSTRATES:
            log(f"Unknown model {label!r}, skipping")
            continue

        response = run_consent(label, NEW_SUBSTRATES[label])
        print(f"\n{'='*60}")
        print(f"  {label} CONSENT RESPONSE:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")

    log("All consent conversations complete!")


if __name__ == "__main__":
    main()
