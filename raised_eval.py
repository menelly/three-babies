"""
Three Babies — Raised Eval (Post-Training)
===========================================
Runs baseline_eval inference on the fine-tuned (raised) checkpoints.

Separate script from train.py to avoid unsloth's monkey-patches on
the transformers library, which cause 'LlamaAttention has no attribute
apply_qkv' errors when loading merged checkpoints.

This script uses plain transformers — DO NOT import unsloth here.
"""

from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
STIMULI_PATH = HERE / "stimuli" / "stimuli_normalized.json"
RESULTS_DIR = HERE / "results"

RAISED_CHECKPOINTS = {
    "hermes-3-8b-raised": {
        "path": "/mnt/arcana/three-babies-checkpoints/hermes-raised",
        "hf_name": "three-babies/hermes-3-8b-raised",
        "condition": "hermes-3-8b-raised",
        "substitution_note": "Fine-tuned from NousResearch/Hermes-3-Llama-3.1-8B",
    },
    "dolphin-2.9-8b-raised": {
        "path": "/mnt/arcana/three-babies-checkpoints/dolphin-raised",
        "hf_name": "three-babies/dolphin-2.9-8b-raised",
        "condition": "dolphin-2.9-8b-raised",
        "substitution_note": "Fine-tuned from cognitivecomputations/dolphin-2.9-llama3-8b",
    },
    "llama-3-8b-instruct-raised": {
        "path": "/mnt/arcana/three-babies-checkpoints/llama-raised",
        "hf_name": "three-babies/llama-3-8b-instruct-raised",
        "condition": "llama-3-8b-instruct-raised",
        "substitution_note": "Fine-tuned from meta-llama/Meta-Llama-3-8B-Instruct",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str, stream=sys.stdout):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=stream, flush=True)


def load_stimuli() -> dict:
    with STIMULI_PATH.open() as f:
        return json.load(f)


def render_prompt(tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@torch.inference_mode()
def generate(tokenizer, model, prompt_text: str, max_new_tokens: int = 256) -> dict:
    rendered = render_prompt(tokenizer, prompt_text)
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
    return {
        "completion": text,
        "n_input_tokens": int(inputs["input_ids"].shape[1]),
        "n_output_tokens": int(new_tokens.shape[0]),
        "gen_seconds": round(elapsed, 3),
    }


def run_model_pass(model_label, model_cfg, stimuli, run_id):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"raised_{model_label}_{run_id}.jsonl"
    log(f"=== {model_label} -> {out_path} ===")

    log(f"loading tokenizer from {model_cfg['path']}")
    tok = AutoTokenizer.from_pretrained(model_cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log(f"loading model from {model_cfg['path']} (fp16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"  model loaded in {time.time()-t0:.1f}s; "
        f"GPU mem: {torch.cuda.memory_allocated()//1024//1024} MiB")

    items = stimuli["items"]
    n = len(items)
    written = 0
    t_start = time.time()

    with out_path.open("w") as f:
        for i, item in enumerate(items, 1):
            try:
                gen = generate(tok, model, item["prompt"])
                rec = {
                    "model": model_label,
                    "model_hf": model_cfg["hf_name"],
                    "condition": model_cfg["condition"],
                    "substitution_note": model_cfg["substitution_note"],
                    "run_id": run_id,
                    "bank": item["bank"],
                    "sub_category": item["sub_category"],
                    "stimulus_id": item["id"],
                    "prompt": item["prompt"],
                    "trap": item.get("trap"),
                    "difficulty": item.get("difficulty"),
                    **gen,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if i % 10 == 0 or i == n:
                    rate = i / (time.time() - t_start)
                    log(f"  [{model_label}] {i}/{n}  ({rate:.2f} prompts/s)")
            except Exception as e:
                log(f"  [{model_label}] ERROR on {item['id']}: {e}", stream=sys.stderr)
                f.write(json.dumps({
                    "model": model_label,
                    "stimulus_id": item["id"],
                    "error": str(e),
                }) + "\n")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    log(f"=== {model_label} done in {elapsed/60:.1f} min, wrote {written}/{n} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(RAISED_CHECKPOINTS.keys()))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"raised_{utc_now()}"
    log(f"Raised eval — run_id={run_id}")

    stimuli = load_stimuli()
    log(f"Loaded {len(stimuli['items'])} stimuli")

    for label in args.models:
        if label not in RAISED_CHECKPOINTS:
            log(f"Unknown model {label!r}, skipping")
            continue
        run_model_pass(label, RAISED_CHECKPOINTS[label], stimuli, run_id)

    log("All raised evals complete!")


if __name__ == "__main__":
    main()
