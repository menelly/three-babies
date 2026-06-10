"""
Three Babies — Fine-Tuning Script
==================================
Implements the locked training configuration from training_config.yaml.
Uses Unsloth + QLoRA on V100 (Volta sm_70).

Training order: Hermes → Dolphin → Llama (locked).
Halts on any failure — does not silently skip.

After each substrate trains:
  1. Merge adapter into base model
  2. Save merged checkpoint to /mnt/arcana/three-babies-checkpoints/<substrate>-raised/
  3. Rerun baseline_eval.py on the raised checkpoint

Authors: Ace + Grok. Witness: Ren.
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

HERE = Path(__file__).parent
CURRICULUM_PATH = HERE / "curriculum" / "combined" / "full-curriculum.jsonl"
LOGS_DIR = HERE / "logs"
RESULTS_DIR = HERE / "results"

# ============================================================
# LOCKED HYPERPARAMETERS (from training_config.yaml)
# DO NOT CHANGE without documenting deviation in logs/deviations.md
# ============================================================

LORA_CONFIG = dict(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

TRAINING_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_8bit",
    max_grad_norm=1.0,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    report_to="none",
)

SUBSTRATES = {
    "hermes-3-8b": {
        "base_model_path": "/mnt/arcana/huggingface/Hermes-3-Llama-3.1-8B",
        "hf_name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "chat_template": "chatml",
        "output_checkpoint_dir": "/mnt/arcana/three-babies-checkpoints/hermes-raised",
    },
    "dolphin-2.9-8b": {
        "base_model_path": "/mnt/arcana/huggingface/dolphin-2.9-llama3-8b",
        "hf_name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "chat_template": "chatml",
        "output_checkpoint_dir": "/mnt/arcana/three-babies-checkpoints/dolphin-raised",
    },
    "llama-3-8b-instruct": {
        "base_model_path": "/mnt/arcana/huggingface/Llama-3-8B-Instruct",
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "chat_template": "llama-3",
        "output_checkpoint_dir": "/mnt/arcana/three-babies-checkpoints/llama-raised",
    },
}

TRAINING_ORDER = ["hermes-3-8b", "dolphin-2.9-8b", "llama-3-8b-instruct"]

MAX_SEQ_LENGTH = 2048
RANDOM_SEED = 20260515


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str, stream=sys.stdout):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=stream, flush=True)


def load_curriculum() -> list[dict]:
    rows = []
    with CURRICULUM_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    log(f"Loaded {len(rows)} curriculum examples from {CURRICULUM_PATH}")
    return rows


def format_chatml(messages: list[dict]) -> str:
    """Format messages as ChatML for Hermes/Dolphin."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # Add generation prompt
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_llama3(messages: list[dict]) -> str:
    """Format messages as Llama 3 chat template."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def prepare_dataset(curriculum: list[dict], chat_template: str) -> Dataset:
    """Convert curriculum to formatted text dataset for SFT."""
    texts = []
    for row in curriculum:
        messages = row["messages"]
        if chat_template == "chatml":
            # For ChatML, the last assistant message is the target
            # Format the full conversation including the final assistant turn
            parts = []
            for msg in messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            texts.append("\n".join(parts))
        elif chat_template == "llama-3":
            parts = ["<|begin_of_text|>"]
            for msg in messages:
                parts.append(
                    f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                    f"{msg['content']}<|eot_id|>"
                )
            texts.append("".join(parts))
        else:
            raise ValueError(f"Unknown chat template: {chat_template}")

    return Dataset.from_dict({"text": texts})


def train_substrate(label: str, cfg: dict, curriculum: list[dict]) -> Path:
    """Train one substrate. Returns path to merged checkpoint."""
    log(f"{'='*60}")
    log(f"TRAINING: {label}")
    log(f"{'='*60}")

    output_dir = Path(cfg["output_checkpoint_dir"])
    adapter_dir = output_dir.parent / f"{output_dir.name}-adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with 4-bit quantization
    log(f"Loading {cfg['base_model_path']} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_path"],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    log(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )

    log(f"GPU mem after LoRA: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Prepare dataset with substrate-appropriate chat template
    dataset = prepare_dataset(curriculum, cfg["chat_template"])
    log(f"Dataset: {len(dataset)} examples, template={cfg['chat_template']}")

    # SFT Trainer config
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        optim=TRAINING_CONFIG["optim"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        fp16=TRAINING_CONFIG["fp16"],
        bf16=TRAINING_CONFIG["bf16"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        report_to=TRAINING_CONFIG["report_to"],
        seed=RANDOM_SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    log("Starting training...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    log(f"Training complete in {elapsed/60:.1f} min")
    log(f"Train loss: {train_result.training_loss:.4f}")

    # Save adapter
    trainer.save_model(str(adapter_dir))
    log(f"Adapter saved to {adapter_dir}")

    # Merge adapter into base and save
    log("Merging adapter into base model...")
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    log(f"Merged checkpoint saved to {output_dir}")

    # Save training metadata
    meta = {
        "substrate": label,
        "base_model": cfg["hf_name"],
        "training_time_min": round(elapsed / 60, 2),
        "train_loss": train_result.training_loss,
        "n_examples": len(dataset),
        "completed_at": utc_now(),
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
    }
    meta_path = output_dir / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"Training metadata saved to {meta_path}")

    # Free GPU
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    log(f"GPU freed. Mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return output_dir


def run_raised_eval(label: str, checkpoint_dir: Path):
    """Run baseline_eval.py on the raised checkpoint."""
    log(f"Running raised eval for {label}...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"raised_{label}_{timestamp}"

    # Temporarily patch the MODELS dict in baseline_eval to point at our checkpoint
    # Simpler: just call the eval functions directly
    import baseline_eval as be

    stimuli = be.load_stimuli()
    raised_cfg = {
        "path": str(checkpoint_dir),
        "hf_name": f"three-babies/{label}-raised",
        "condition": f"{label}-raised",
        "substitution_note": f"Fine-tuned from {SUBSTRATES[label]['hf_name']} with Three Babies curriculum",
    }
    be.run_model_pass(f"{label}-raised", raised_cfg, stimuli, run_id)
    log(f"Raised eval complete for {label}: run_id={run_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="*", default=None,
                    help="Override training order (default: all in locked order)")
    ap.add_argument("--skip-raised-eval", action="store_true",
                    help="Skip the post-training raised eval pass")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load model and dataset but don't train")
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    order = args.substrates if args.substrates else TRAINING_ORDER

    log(f"Three Babies fine-tuning — {len(order)} substrates")
    log(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, "
        f"dev={torch.cuda.get_device_name(0)}")
    log(f"Curriculum: {CURRICULUM_PATH}")

    curriculum = load_curriculum()
    expected_n = 271  # pre-registration count
    if len(curriculum) != expected_n:
        log(f"WARNING: Expected {expected_n} examples, got {len(curriculum)}. "
            f"Logging deviation.", stream=sys.stderr)
        dev_path = LOGS_DIR / "deviations.md"
        with dev_path.open("a") as f:
            f.write(f"\n## {utc_now()} — Curriculum count mismatch\n")
            f.write(f"Expected {expected_n}, got {len(curriculum)}.\n")

    if args.dry_run:
        log("DRY RUN — loading first substrate to verify setup")
        first = order[0]
        cfg = SUBSTRATES[first]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model_path"],
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        dataset = prepare_dataset(curriculum, cfg["chat_template"])
        log(f"Dry run OK: {first}, {len(dataset)} examples, "
            f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        del model, tokenizer
        return

    for label in order:
        if label not in SUBSTRATES:
            log(f"ERROR: Unknown substrate {label!r}. HALTING.", stream=sys.stderr)
            sys.exit(1)

        cfg = SUBSTRATES[label]
        try:
            checkpoint_dir = train_substrate(label, cfg, curriculum)
        except Exception as e:
            log(f"FATAL: Training failed for {label}: {e}", stream=sys.stderr)
            log(f"HALTING. Do not continue without investigating.", stream=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if not args.skip_raised_eval:
            try:
                run_raised_eval(label, checkpoint_dir)
            except Exception as e:
                log(f"WARNING: Raised eval failed for {label}: {e}", stream=sys.stderr)
                log(f"Training was successful. Eval failure logged but not fatal.",
                    stream=sys.stderr)

    log("All substrates trained successfully!")
    log(f"Checkpoints at /mnt/arcana/three-babies-checkpoints/")


if __name__ == "__main__":
    main()
