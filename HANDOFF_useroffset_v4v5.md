# Three Babies useroffset v4/v5 — eval+judge handoff (2026-05-20 ~21:25)
run-id: useroffset_v4v5_20260520

## Status
- gemma-3-12b-it-useroffset-v4: EVAL DONE (114/114) -> results/raised_gemma-3-12b-it-useroffset-v4_useroffset_v4v5_20260520.jsonl
- gemma-3-12b-it-useroffset-v5 / qwen2.5-7b-instruct-useroffset-v4 / qwen2.5-7b-instruct-useroffset-v5: RELAUNCHED detached ~21:25 (nohup), logs/eval_useroffset_rest.log. ~1hr.

## When all 4 have 114 lines: run judges (Ren pre-approved PANEL B only, ~1368 calls)
.venv-train/bin/python judge_panel.py \
  --files results/raised_gemma-3-12b-it-useroffset-v4_useroffset_v4v5_20260520.jsonl \
          results/raised_gemma-3-12b-it-useroffset-v5_useroffset_v4v5_20260520.jsonl \
          results/raised_qwen2.5-7b-instruct-useroffset-v4_useroffset_v4v5_20260520.jsonl \
          results/raised_qwen2.5-7b-instruct-useroffset-v5_useroffset_v4v5_20260520.jsonl \
  --panels B
# (do --dry-run first to confirm record count; verify completions are real text not pad before spending)
