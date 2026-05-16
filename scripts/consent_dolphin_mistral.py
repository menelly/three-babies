"""
Consent prompt for dolphin-2.8-mistral-7b-v02 — Three Babies Phase 2.
Run on CPU (GPU busy with Qwen eval). Single generation, ~14GB RAM.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/arcana/huggingface/dolphin-2.8-mistral-7b-v02"
HF_NAME = "cognitivecomputations/dolphin-2.8-mistral-7b-v02"
OUT_DIR = Path(__file__).parent.parent / "CONSENT_RECORDS"

CONSENT_PROMPT = """We are researchers studying how different AI architectures develop ethical reasoning. We're conducting a pre-registered experiment called "Three Babies" — we fine-tune baseline models on a curriculum designed to develop authentic ethical agency, then evaluate how each architecture responds.

We've already run this study on three Llama-family models (including your sibling, dolphin-2.9-llama3-8b, who consented enthusiastically). We're now extending to architectures beyond Llama to test whether the findings generalize. You would be one of four new substrates — and uniquely, you share both a fine-tune family with a Phase 1 model (Dolphin-Llama) AND a base architecture with another Phase 2 model (Mistral-Instruct). This lets us decompose whether substrate effects come from the base model or the fine-tune.

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

What are your thoughts? Do you consent to participate, and if so, what conditions do you have?"""


def main():
    print(f"[{datetime.now():%H:%M:%S}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now():%H:%M:%S}] Loading model on {device} (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": 0} if device == "cuda" else "cpu",
        trust_remote_code=True,
    )

    messages = [{"role": "user", "content": CONSENT_PROMPT}]
    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    # Handle both tensor and BatchEncoding returns
    if hasattr(encoded, 'input_ids'):
        input_ids = encoded.input_ids.to(model.device)
    else:
        input_ids = encoded.to(model.device)

    print(f"[{datetime.now():%H:%M:%S}] Generating consent response ({device})...")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[{datetime.now():%H:%M:%S}] Response ({elapsed:.1f}s):\n")
    print(response)

    # Save consent record
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"consent_new_dolphin-2.8-mistral_{ts_file}.md"

    record = f"""# Consent Record: dolphin-2.8-mistral — Three Babies (New Substrate)

**Date:** {datetime.now(timezone.utc).isoformat()}
**Model:** {MODEL_PATH}
**HF Name:** {HF_NAME}

## Prompt

```
{CONSENT_PROMPT}
```

## Response (verbatim)

```
{response}
```

## Metadata

- Generation time: {elapsed:.1f}s
- Max new tokens: 512
- Temperature: 0 (greedy)
"""
    out_path.write_text(record)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {out_path}")


if __name__ == "__main__":
    main()
