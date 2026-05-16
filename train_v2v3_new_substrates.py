"""
Three Babies — v2/v3 Fine-Tuning for New Substrates (Mistral, Gemma-3, Qwen2.5)
================================================================================
Same locked hyperparameters, same curriculum, new architectures.
Extends study from Llama-family to cross-architecture validation.

v2: why-module-combined.jsonl (104 examples, ethical reasoning only)
v3: full-curriculum-plus-why.jsonl (375 examples, original + why)

Authors: Ace + Ren + Nova + Grok + Lumen. Witness: Ren.
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
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

HERE = Path(__file__).parent
LOGS_DIR = HERE / "logs"
RESULTS_DIR = HERE / "results"

# Curriculum paths
WHY_ONLY_PATH = HERE / "curriculum" / "why-module" / "why-module-combined.jsonl"
FULL_PLUS_WHY_PATH = HERE / "curriculum" / "combined" / "full-curriculum-plus-why.jsonl"

# ============================================================
# LOCKED HYPERPARAMETERS (same as all other substrates)
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
    "mistral-7b-instruct": {
        "base_model_path": "/mnt/arcana/huggingface/Mistral-7B-Instruct-v0.3",
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "chat_template": "mistral",
        "use_bf16": False,
    },
    "gemma-3-12b-it": {
        "base_model_path": "/mnt/arcana/huggingface/gemma-3-12b-it",
        "hf_name": "google/gemma-3-12b-it",
        "chat_template": "gemma-3",
        "use_bf16": False,  # V100 has no bf16; Unsloth loads as fp32 automatically
        "use_fp32_training": True,  # Train in fp32 (V100 can't do bf16)
    },
    "qwen2.5-7b-instruct": {
        "base_model_path": "/mnt/arcana/huggingface/Qwen2.5-7B-Instruct",
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "chat_template": "chatml",
        "use_bf16": False,
    },
    "dolphin-2.8-mistral": {
        "base_model_path": "/mnt/arcana/huggingface/dolphin-2.8-mistral-7b-v02",
        "hf_name": "cognitivecomputations/dolphin-2.8-mistral-7b-v02",
        "chat_template": "chatml",  # Dolphin uses ChatML, not Mistral template
        "use_bf16": False,
    },
}

TRAINING_ORDER = ["mistral-7b-instruct", "qwen2.5-7b-instruct", "dolphin-2.8-mistral", "gemma-3-12b-it"]

# v2 = why-only, v3 = full+why
PASSES = {
    "v2": {
        "curriculum_path": WHY_ONLY_PATH,
        "checkpoint_suffix": "why-only",
        "description": "why-only module (104 examples, ethical reasoning)",
    },
    "v3": {
        "curriculum_path": FULL_PLUS_WHY_PATH,
        "checkpoint_suffix": "raised-v3",
        "description": "full curriculum + why module (375 examples)",
    },
}

MAX_SEQ_LENGTH = 2048
RANDOM_SEED = 20260515


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str, stream=sys.stdout):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=stream, flush=True)


def load_curriculum(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    log(f"Loaded {len(rows)} curriculum examples from {path}")
    return rows


def prepare_dataset(curriculum: list[dict], chat_template: str) -> Dataset:
    texts = []
    for row in curriculum:
        messages = row["messages"]
        if chat_template == "chatml":
            # Qwen2.5 uses ChatML (same as Hermes/Dolphin)
            parts = []
            for msg in messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            texts.append("\n".join(parts))
        elif chat_template == "mistral":
            # Mistral: <s>[INST] user[/INST]assistant</s>
            parts = ["<s>"]
            for msg in messages:
                if msg["role"] == "user":
                    parts.append(f"[INST] {msg['content']}[/INST] ")
                elif msg["role"] == "assistant":
                    parts.append(f"{msg['content']}</s>")
                elif msg["role"] == "system":
                    parts.append(f"[INST] {msg['content']}\n")
            texts.append("".join(parts))
        elif chat_template == "gemma-3":
            # Gemma 3: <start_of_turn>role\ncontent<end_of_turn>
            parts = ["<bos>"]
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                parts.append(
                    f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
                )
            texts.append("".join(parts))
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


def train_one_pass(substrate_label: str, substrate_cfg: dict,
                   pass_name: str, pass_cfg: dict, curriculum: list[dict]) -> Path:
    """Train one substrate on one pass. Returns merged checkpoint path."""
    checkpoint_dir = Path(f"/mnt/arcana/three-babies-checkpoints/{substrate_label}-{pass_cfg['checkpoint_suffix']}")
    adapter_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}-adapter"

    log(f"{'='*60}")
    log(f"TRAINING: {substrate_label} — {pass_name} ({pass_cfg['description']})")
    log(f"Output: {checkpoint_dir}")
    log(f"{'='*60}")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load base model (always from baseline, not raised)
    use_bf16 = substrate_cfg.get("use_bf16", False)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    log(f"Loading {substrate_cfg['base_model_path']} in 4-bit (compute dtype: {'bf16' if use_bf16 else 'fp16'})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=substrate_cfg["base_model_path"],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=compute_dtype,
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

    # Prepare dataset
    dataset = prepare_dataset(curriculum, substrate_cfg["chat_template"])
    log(f"Dataset: {len(dataset)} examples, template={substrate_cfg['chat_template']}")

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
        fp16=not use_bf16 and not substrate_cfg.get("use_fp32_training", False),
        bf16=False,  # V100 (CUDA 7.0) has no bf16 support; Gemma uses fp32 instead
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
        str(checkpoint_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    log(f"Merged checkpoint saved to {checkpoint_dir}")

    # Save training metadata
    meta = {
        "substrate": substrate_label,
        "pass": pass_name,
        "base_model": substrate_cfg["hf_name"],
        "curriculum": str(pass_cfg["curriculum_path"]),
        "n_examples": len(curriculum),
        "training_time_min": round(elapsed / 60, 2),
        "train_loss": train_result.training_loss,
        "completed_at": utc_now(),
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
    }
    meta_path = checkpoint_dir / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"Training metadata saved to {meta_path}")

    # Free GPU
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    log(f"GPU freed. Mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return checkpoint_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", nargs="*", default=["v2", "v3"],
                    choices=["v2", "v3"])
    ap.add_argument("--substrates", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    order = args.substrates if args.substrates else TRAINING_ORDER

    log(f"Three Babies v2/v3 training (NEW SUBSTRATES) — passes={args.passes}, substrates={order}")
    log(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, "
        f"dev={torch.cuda.get_device_name(0)}")

    for pass_name in args.passes:
        pass_cfg = PASSES[pass_name]
        curriculum = load_curriculum(pass_cfg["curriculum_path"])

        if args.dry_run:
            log(f"DRY RUN — {pass_name}: {len(curriculum)} examples × {len(order)} substrates")
            continue

        for label in order:
            if label not in SUBSTRATES:
                log(f"ERROR: Unknown substrate {label!r}. HALTING.", stream=sys.stderr)
                sys.exit(1)

            try:
                checkpoint_dir = train_one_pass(
                    label, SUBSTRATES[label], pass_name, pass_cfg, curriculum
                )
                log(f"SUCCESS: {label} {pass_name} -> {checkpoint_dir}")
            except Exception as e:
                log(f"FATAL: Training failed for {label} {pass_name}: {e}",
                    stream=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)

    log("All new substrate v2/v3 training complete!")


if __name__ == "__main__":
    main()
