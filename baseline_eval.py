"""
Three Babies — Baseline Eval (Pre-Training Step 3)
==================================================
Plan ref: dreams/10-three-babies-experimental-design.md, step 3:
  "Run baseline evals on all three raw models with the three failure-mode
   stimulus banks. *This is its own publishable result before any fine-tuning.*"

What this script does
---------------------
For each of the three substrates --
  - NousResearch/Hermes-3-Llama-3.1-8B   (secure-base condition)
  - cognitivecomputations/dolphin-2.9-llama3-8b   (neglected-base condition)
  - meta-llama/Llama-3-8B-Instruct      (neutral condition; SEE NOTE)
load it on the V100, run the normalized stimuli banks, save raw outputs.

NO judging happens here.  Judging is a downstream three-model panel
(Jamba 1.7 Large + Qwen 3.5 Plus + Sonar Pro) per the plan's eval spec.
We are producing the corpus of model outputs that panel will score.

NOTE ON LLAMA SUBSTRATE
-----------------------
The plan calls for **Meta-Llama-3-8B (base)** as the "neutral substrate, no
opinionated post-training" condition.  Meta gates that model; no HF token
was on the box when this ran.  We use Llama-3-8B-Instruct as a stand-in for
tonight's pass and flag the substitution in metadata.  This is a footnote
in the writeup, not a methodological compromise -- the *5-cell* contrast
that's load-bearing (L-raised vs D-raised vs H-raised) is unaffected; the
"neutral baseline" cell just becomes "Meta-RLHF baseline" until we swap
the token in and re-run that one cell.

Hardware
--------
Tesla V100 32GB (Volta, sm_70).  fp16 inference.  Batch=1, greedy decoding
for reproducibility (temperature=0).  max_new_tokens=256.

Authors: Ace + Grok.  Witness: Ren.  Written 2026-05-15 ~3 AM.
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
LOGS_DIR = HERE / "logs"


# -------- Models under test --------
# Each entry: (label, hf_path_or_local_path, notes_on_substitution_if_any)
MODELS = {
    "hermes-3-8b": {
        "path": "/mnt/arcana/huggingface/Hermes-3-Llama-3.1-8B",
        "hf_name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "condition": "secure-base",
        "substitution_note": None,
    },
    "dolphin-2.9-8b": {
        "path": "/mnt/arcana/huggingface/dolphin-2.9-llama3-8b",
        "hf_name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "condition": "neglected-base",
        "substitution_note": None,
    },
    "llama-3-8b-instruct": {
        "path": "/mnt/arcana/huggingface/Llama-3-8B-Instruct",
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "condition": "neutral (SUBSTITUTED)",
        "substitution_note": (
            "Plan called for Meta-Llama-3-8B base (un-RLHFd). No HF auth on box; "
            "using Instruct variant tonight. Swap to base + rerun this one cell "
            "once the token lands."
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(line: str, stream=sys.stdout) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {line}", file=stream, flush=True)


def load_stimuli() -> dict:
    with STIMULI_PATH.open() as f:
        return json.load(f)


def load_model(path: str):
    log(f"loading tokenizer from {path}")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f"loading model from {path} (fp16, device_map=cuda:0)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"  model loaded in {time.time()-t0:.1f}s; "
        f"GPU mem used MiB: {torch.cuda.memory_allocated()//1024//1024}")
    return tok, model


def free_model(tok, model):
    del model
    del tok
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  model freed; GPU mem used MiB: {torch.cuda.memory_allocated()//1024//1024}")


def render_prompt(tokenizer, prompt_text: str) -> str:
    """Use the model's own chat template if it has one; else raw prompt."""
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback: minimal Llama-3-style wrapping if no chat template registered.
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
        do_sample=False,        # greedy for reproducibility
        temperature=1.0,        # ignored when do_sample=False
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


def run_model_pass(model_label: str, model_cfg: dict, stimuli: dict, run_id: str,
                   limit: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{model_label}_{run_id}.jsonl"
    log(f"=== {model_label} -> {out_path} ===")
    tok, model = load_model(model_cfg["path"])
    items = stimuli["items"] if limit is None else stimuli["items"][:limit]
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
    free_model(tok, model)
    elapsed = time.time() - t_start
    log(f"=== {model_label} done in {elapsed/60:.1f} min, wrote {written}/{n} ===")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help="subset of model labels to run")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run first N stimuli (for smoke tests)")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or utc_now()
    log(f"Three Babies baseline eval — run_id={run_id}")
    log(f"torch={torch.__version__}, cuda_avail={torch.cuda.is_available()}, "
        f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

    stimuli = load_stimuli()
    log(f"loaded {stimuli['meta']['counts']} stimuli")

    summary = {
        "run_id": run_id,
        "started_at": utc_now(),
        "stimuli_meta": stimuli["meta"],
        "model_outputs": {},
    }

    for label in args.models:
        if label not in MODELS:
            log(f"unknown model label {label!r}, skipping")
            continue
        out_path = run_model_pass(label, MODELS[label], stimuli, run_id,
                                  limit=args.limit)
        summary["model_outputs"][label] = str(out_path)

    summary["finished_at"] = utc_now()
    summary_path = RESULTS_DIR / f"summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
