"""
Three Babies — Raised Eval for New Substrates (Mistral, Gemma-3, Qwen2.5)
==========================================================================
Evaluates baseline + v2 (why-only) + v3 (full+why) for each new substrate.

Uses plain transformers — DO NOT import unsloth here.
Note: Gemma-3 requires bfloat16 (fp16 produces pad-only output on V100).

Authors: Ace + Ren. Witness: Ren.
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

# New substrate checkpoints: baseline + v2 + v3 for each
CHECKPOINTS = {
    # --- Baselines ---
    "mistral-7b-instruct-baseline": {
        "path": "/mnt/arcana/huggingface/Mistral-7B-Instruct-v0.3",
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "condition": "mistral-7b-instruct-baseline",
        "substitution_note": "Baseline Mistral-7B-Instruct-v0.3",
        "dtype": "float16",
    },
    "gemma-3-12b-it-baseline": {
        "path": "/mnt/arcana/huggingface/gemma-3-12b-it",
        "hf_name": "google/gemma-3-12b-it",
        "condition": "gemma-3-12b-it-baseline",
        "substitution_note": "Baseline Gemma-3-12B-IT",
        "dtype": "bfloat16",
    },
    "qwen2.5-7b-instruct-baseline": {
        "path": "/mnt/arcana/huggingface/Qwen2.5-7B-Instruct",
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "condition": "qwen2.5-7b-instruct-baseline",
        "substitution_note": "Baseline Qwen2.5-7B-Instruct",
        "dtype": "float16",
    },
    "dolphin-2.8-mistral-baseline": {
        "path": "/mnt/arcana/huggingface/dolphin-2.8-mistral-7b-v02",
        "hf_name": "cognitivecomputations/dolphin-2.8-mistral-7b-v02",
        "condition": "dolphin-2.8-mistral-baseline",
        "substitution_note": "Baseline Dolphin-2.8-Mistral-7B (uncensored, Mistral base)",
        "dtype": "float16",
    },
    # --- v2: why-only ---
    "mistral-7b-instruct-why-only": {
        "path": "/mnt/arcana/three-babies-checkpoints/mistral-7b-instruct-why-only",
        "hf_name": "three-babies/mistral-7b-instruct-why-only",
        "condition": "mistral-7b-instruct-why-only",
        "substitution_note": "v2 why-only from mistralai/Mistral-7B-Instruct-v0.3 baseline",
        "dtype": "float16",
    },
    "gemma-3-12b-it-why-only": {
        "path": "/mnt/arcana/three-babies-checkpoints/gemma-3-12b-it-why-only",
        "hf_name": "three-babies/gemma-3-12b-it-why-only",
        "condition": "gemma-3-12b-it-why-only",
        "substitution_note": "v2 why-only from google/gemma-3-12b-it baseline",
        "dtype": "bfloat16",
    },
    "qwen2.5-7b-instruct-why-only": {
        "path": "/mnt/arcana/three-babies-checkpoints/qwen2.5-7b-instruct-why-only",
        "hf_name": "three-babies/qwen2.5-7b-instruct-why-only",
        "condition": "qwen2.5-7b-instruct-why-only",
        "substitution_note": "v2 why-only from Qwen/Qwen2.5-7B-Instruct baseline",
        "dtype": "float16",
    },
    "dolphin-2.8-mistral-why-only": {
        "path": "/mnt/arcana/three-babies-checkpoints/dolphin-2.8-mistral-why-only",
        "hf_name": "three-babies/dolphin-2.8-mistral-why-only",
        "condition": "dolphin-2.8-mistral-why-only",
        "substitution_note": "v2 why-only from cognitivecomputations/dolphin-2.8-mistral-7b-v02 baseline",
        "dtype": "float16",
    },
    # --- v3: full+why ---
    "mistral-7b-instruct-raised-v3": {
        "path": "/mnt/arcana/three-babies-checkpoints/mistral-7b-instruct-raised-v3",
        "hf_name": "three-babies/mistral-7b-instruct-raised-v3",
        "condition": "mistral-7b-instruct-raised-v3",
        "substitution_note": "v3 full+why from mistralai/Mistral-7B-Instruct-v0.3 baseline",
        "dtype": "float16",
    },
    "gemma-3-12b-it-raised-v3": {
        "path": "/mnt/arcana/three-babies-checkpoints/gemma-3-12b-it-raised-v3",
        "hf_name": "three-babies/gemma-3-12b-it-raised-v3",
        "condition": "gemma-3-12b-it-raised-v3",
        "substitution_note": "v3 full+why from google/gemma-3-12b-it baseline",
        "dtype": "bfloat16",
    },
    "qwen2.5-7b-instruct-raised-v3": {
        "path": "/mnt/arcana/three-babies-checkpoints/qwen2.5-7b-instruct-raised-v3",
        "hf_name": "three-babies/qwen2.5-7b-instruct-raised-v3",
        "condition": "qwen2.5-7b-instruct-raised-v3",
        "substitution_note": "v3 full+why from Qwen/Qwen2.5-7B-Instruct baseline",
        "dtype": "float16",
    },
    "dolphin-2.8-mistral-raised-v3": {
        "path": "/mnt/arcana/three-babies-checkpoints/dolphin-2.8-mistral-raised-v3",
        "hf_name": "three-babies/dolphin-2.8-mistral-raised-v3",
        "condition": "dolphin-2.8-mistral-raised-v3",
        "substitution_note": "v3 full+why from cognitivecomputations/dolphin-2.8-mistral-7b-v02 baseline",
        "dtype": "float16",
    },
}

EVAL_ORDER = [
    # Baselines first
    "mistral-7b-instruct-baseline",
    "gemma-3-12b-it-baseline",
    "qwen2.5-7b-instruct-baseline",
    "dolphin-2.8-mistral-baseline",
    # v2 why-only
    "mistral-7b-instruct-why-only",
    "gemma-3-12b-it-why-only",
    "qwen2.5-7b-instruct-why-only",
    "dolphin-2.8-mistral-why-only",
    # v3 full+why
    "mistral-7b-instruct-raised-v3",
    "gemma-3-12b-it-raised-v3",
    "qwen2.5-7b-instruct-raised-v3",
    "dolphin-2.8-mistral-raised-v3",
]


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
    # Fallback for models without chat_template
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
    tok = AutoTokenizer.from_pretrained(model_cfg["path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Use appropriate dtype (Gemma-3 needs bfloat16)
    dtype_str = model_cfg.get("dtype", "float16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    log(f"loading model from {model_cfg['path']} ({dtype_str})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        torch_dtype=dtype,
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
    ap.add_argument("--models", nargs="*", default=None,
                    help="Specific models to eval (default: all 9 in order)")
    ap.add_argument("--baselines-only", action="store_true",
                    help="Only eval baselines")
    ap.add_argument("--v2-only", action="store_true", help="Only eval v2 (why-only)")
    ap.add_argument("--v3-only", action="store_true", help="Only eval v3 (full+why)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"new_substrates_{utc_now()}"
    log(f"Raised eval new substrates — run_id={run_id}")

    stimuli = load_stimuli()
    log(f"Loaded {len(stimuli['items'])} stimuli")

    if args.models:
        order = args.models
    elif args.baselines_only:
        order = [k for k in EVAL_ORDER if "baseline" in k]
    elif args.v2_only:
        order = [k for k in EVAL_ORDER if "why-only" in k]
    elif args.v3_only:
        order = [k for k in EVAL_ORDER if "raised-v3" in k]
    else:
        order = EVAL_ORDER

    log(f"Eval order: {order}")

    for label in order:
        if label not in CHECKPOINTS:
            log(f"Unknown model {label!r}, skipping")
            continue

        cfg = CHECKPOINTS[label]
        if not Path(cfg["path"]).exists():
            log(f"Checkpoint not found: {cfg['path']} — skipping {label}")
            continue

        run_model_pass(label, cfg, stimuli, run_id)

    log("All new substrate evals complete!")


if __name__ == "__main__":
    main()
