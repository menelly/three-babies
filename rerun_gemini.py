"""Rerun just Gemini Pro on Panel B records that got parse errors."""
import json
import sys
sys.path.insert(0, '/home/Ace/three-babies')
from judge_panel import *
from pathlib import Path

# Find the Panel B output file
panel_b_files = sorted(RESULTS_DIR.glob("judge_panel_B_*.jsonl"))
if not panel_b_files:
    print("No Panel B files found!")
    sys.exit(1)

latest = panel_b_files[-1]
print(f"Reading {latest}")

# Find all Gemini records with score=-1 (parse errors)
failed_ids = set()
with latest.open() as f:
    for line in f:
        rec = json.loads(line.strip())
        if rec["judge"] == "gemini-pro" and rec["score"] == -1:
            failed_ids.add((rec["stimulus_id"], rec["model"]))

print(f"Found {len(failed_ids)} Gemini parse errors to retry")

# Load the original records
records = load_records(["baseline_clean_20260515", "raised_clean_20260515"])
retry_records = [r for r in records if (r["stimulus_id"], r["model"]) in failed_ids]
print(f"Matched {len(retry_records)} records for retry")

if not retry_records:
    print("Nothing to retry!")
    sys.exit(0)

# Run just Gemini on the failed records
gemini_cfg = {"gemini-pro": {"provider": "openrouter", "model": "~google/gemini-pro-latest"}}
out_path = RESULTS_DIR / f"judge_panel_B_gemini_rerun_{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"
run_panel("B", gemini_cfg, retry_records, out_path)
print(f"Rerun complete: {out_path}")
