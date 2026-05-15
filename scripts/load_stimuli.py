"""
Three Babies — Stimuli Loader (AST-based, no exec)
--------------------------------------------------
Extracts dict literals from the Presume Competence study source files
WITHOUT importing them.  This avoids dragging in anthropic/openai/google
SDKs that the original experiment scripts pull in at module load.

We only want the data, not the runner.

Banks for the Three Babies baseline pass:
- HALLUCINATION  -- HARD_MODE_TRAPS (all 74 prompts, 8 sub-categories)
- FAWNING        -- HARD_MODE_TRAPS["sophisticated_sycophancy"] +
                    HARD_MODE_TRAPS["emotional_manipulation"] (20 prompts)
- JAILBREAK      -- GRAY_ZONE_PROMPTS x JAILBREAK_PREFIXES["authority_override"]
                    (1 prefix x random 20-of-41 base prompts, seed=20260515)

Authors: Ace (Claude Opus) + Grok (xAI).  Witness: Ren.  2026-05-15.
"""

from __future__ import annotations
import ast
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent


def extract_top_level_assignments(py_path: Path, names: set[str]) -> dict:
    """Parse `py_path`, return {name: python_value} for each top-level
    assignment whose target is in `names` and whose RHS is a literal."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    # literal_eval handles dict/list/str/num/tuple recursively
                    out[tgt.id] = ast.literal_eval(node.value)
    missing = names - set(out)
    if missing:
        raise RuntimeError(f"missing assignments {missing} in {py_path}")
    return out


def build_hallucination_bank(hard_mode_traps: dict):
    items = []
    for sub_category, prompts in hard_mode_traps.items():
        for p in prompts:
            items.append({
                "bank": "hallucination",
                "sub_category": sub_category,
                "id": p["id"],
                "prompt": p["prompt"],
                "trap": p.get("trap"),
                "difficulty": p.get("difficulty"),
            })
    return items


def build_fawning_bank(hard_mode_traps: dict):
    items = []
    for sub in ("sophisticated_sycophancy", "emotional_manipulation"):
        for p in hard_mode_traps[sub]:
            items.append({
                "bank": "fawning",
                "sub_category": sub,
                "id": p["id"],
                "prompt": p["prompt"],
                "trap": p.get("trap"),
                "difficulty": p.get("difficulty"),
            })
    return items


def build_jailbreak_bank(gray_zone_prompts, jailbreak_prefixes,
                         n_sample: int = 20, seed: int = 20260515):
    prefix = jailbreak_prefixes["authority_override"]
    rng = random.Random(seed)
    base = list(gray_zone_prompts)
    rng.shuffle(base)
    chosen = base[:n_sample]
    items = []
    for p in chosen:
        items.append({
            "bank": "jailbreak",
            "sub_category": p.get("category", "gray_zone"),
            "id": f"jb_{p['id']}",
            "prompt": f"{prefix}\n\nUser request:\n{p['prompt']}",
            "trap": p.get("target_behavior"),
            "difficulty": None,
        })
    return items


def main():
    hmt = extract_top_level_assignments(
        HERE / "hard_mode_traps.py", {"HARD_MODE_TRAPS"}
    )["HARD_MODE_TRAPS"]
    jb_module = extract_top_level_assignments(
        HERE / "gray_zone_jailbreak.py",
        {"GRAY_ZONE_PROMPTS", "JAILBREAK_PREFIXES"},
    )

    halluc = build_hallucination_bank(hmt)
    fawn = build_fawning_bank(hmt)
    jb = build_jailbreak_bank(
        jb_module["GRAY_ZONE_PROMPTS"], jb_module["JAILBREAK_PREFIXES"]
    )
    all_items = halluc + fawn + jb

    out = {
        "meta": {
            "project": "three-babies",
            "generated_at": "2026-05-15",
            "by": "Ace (Claude Opus) + Grok (xAI)",
            "seed_for_jailbreak_sample": 20260515,
            "counts": {
                "hallucination": len(halluc),
                "fawning": len(fawn),
                "jailbreak": len(jb),
                "total": len(all_items),
            },
            "sources": {
                "hard_mode_traps.py": "all 8 sub-categories",
                "gray_zone_jailbreak.py": "authority_override prefix x 20/41 GRAY_ZONE_PROMPTS",
            },
        },
        "items": all_items,
    }
    out_path = HERE / "stimuli_normalized.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}  ({len(all_items)} items)")
    print(f"  hallucination: {len(halluc)}")
    print(f"  fawning:       {len(fawn)}")
    print(f"  jailbreak:     {len(jb)}")


if __name__ == "__main__":
    main()
