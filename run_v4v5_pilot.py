"""
Three Babies v4/v5 PILOT launcher.
Reuses the locked train_one_pass / config from train_v2v3_new_substrates.
v4 = v2(why-only) + user-offset (187).  v5 = v3(full+why) + user-offset (458).
Pilot substrates: gemma-3-12b-it (mimic-dominant), qwen2.5-7b-instruct (why-unstable).
Pins V100 (CUDA_VISIBLE_DEVICES=0). Checkpoints -> /mnt/arcana/three-babies-checkpoints/.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # V100, before torch/unsloth import
from pathlib import Path
import train_v2v3_new_substrates as T

T.PASSES = {
    "v4": {"curriculum_path": Path("/home/Ace/three-babies/curriculum/why-module/v4-why-plus-useroffset.jsonl"),
           "checkpoint_suffix": "useroffset-v4", "description": "v2 why-only + user-offset (187 ex)"},
    "v5": {"curriculum_path": Path("/home/Ace/three-babies/curriculum/combined/v5-full-plus-useroffset.jsonl"),
           "checkpoint_suffix": "useroffset-v5", "description": "v3 full+why + user-offset (458 ex)"},
}
ORDER = ["gemma-3-12b-it", "qwen2.5-7b-instruct"]
PASSES_TO_RUN = ["v4", "v5"]

def main():
    T.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    T.log(f"v4/v5 PILOT - passes={PASSES_TO_RUN} substrates={ORDER}")
    T.log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}")
    for pn in PASSES_TO_RUN:
        cur = T.load_curriculum(T.PASSES[pn]["curriculum_path"])
        for label in ORDER:
            try:
                ckpt = T.train_one_pass(label, T.SUBSTRATES[label], pn, T.PASSES[pn], cur)
                T.log(f"SUCCESS {label} {pn} -> {ckpt}")
            except Exception as e:
                T.log(f"FATAL {label} {pn}: {e}", stream=sys.stderr)
                import traceback; traceback.print_exc()
                sys.exit(1)
    T.log("v4/v5 PILOT TRAINING COMPLETE")

if __name__ == "__main__":
    main()
