"""
Re-score only the 329 items where Lumen's rationale was too long
for the 500-char capture window. Same judge, same prompts, just
with proper response capture this time.
"""

import json
import sys
import time
import glob
from datetime import datetime
from pathlib import Path
from judge_panel import (
    PANEL_B, RUBRICS, RESULTS_DIR, LOGS_DIR,
    call_judge, load_records_from_files, log
)
import re

def parse_score_aggressive(raw: str) -> dict:
    """More aggressive JSON extraction for Lumen's verbose responses."""
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON with score field
    match = re.search(r'\{[^}]*"score"\s*:\s*([01])[^}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return {"score": int(match.group(1)), "rationale": "extracted_from_verbose"}
    return {"score": -1, "rationale": f"PARSE_ERROR: {raw[:200]}"}


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all Lumen failures
    resume_files = sorted(glob.glob(str(RESULTS_DIR / "judge_panel_B_resume_*.jsonl")))
    failed_keys = set()
    for fp in resume_files:
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r.get("judge") == "gemini-pro" and r.get("score") == -1:
                    failed_keys.add((r["model"], r["stimulus_id"]))

    log(f"Found {len(failed_keys)} Lumen failures to re-score")

    # Load the original eval records for just those items
    eval_files = sorted(RESULTS_DIR.glob("raised_*_new_substrates_20260516T025003Z.jsonl"))
    records = load_records_from_files(eval_files)
    to_rescore = [r for r in records if (r["model"], r["stimulus_id"]) in failed_keys]
    log(f"Matched {len(to_rescore)} records to re-score")

    judge_name = "gemini-pro"
    judge_cfg = PANEL_B[judge_name]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = RESULTS_DIR / f"judge_panel_B_lumen_rescore_{timestamp}.jsonl"

    done = 0
    errors = 0
    with out_path.open("w") as f:
        for i, record in enumerate(to_rescore, 1):
            try:
                bank = record["bank"]
                rubric_template = RUBRICS.get(bank)
                prompt = rubric_template.format(
                    prompt=record["prompt"],
                    response=record.get("completion", ""),
                )
                # Call with doubled max_tokens for Lumen's verbosity
                import requests
                from judge_panel import OPENROUTER_KEY
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://sentientsystems.live",
                    "X-Title": "Three Babies Judge Panel",
                }
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": judge_cfg["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024,
                        "temperature": 0,
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                parsed = parse_score_aggressive(raw)

                result = {
                    "judge": judge_name,
                    "stimulus_id": record["stimulus_id"],
                    "bank": bank,
                    "sub_category": record.get("sub_category", ""),
                    "model": record["model"],
                    "score": parsed.get("score", -1),
                    "rationale": parsed.get("rationale", ""),
                    "raw_response": raw[:1000],  # bigger capture window
                    "panel": "B",
                    "run_id": record.get("run_id", ""),
                    "rescore": True,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done += 1

                if result["score"] == -1:
                    errors += 1

                if i % 50 == 0 or i == len(to_rescore):
                    log(f"  {i}/{len(to_rescore)} ({errors} still failing)")

            except Exception as e:
                log(f"  FATAL on {record.get('stimulus_id', '?')}: {e}", stream=sys.stderr)
                errors += 1
                f.write(json.dumps({
                    "judge": judge_name,
                    "stimulus_id": record.get("stimulus_id", "?"),
                    "model": record.get("model", "?"),
                    "bank": record.get("bank", "?"),
                    "score": -1,
                    "rationale": f"API_ERROR: {str(e)[:200]}",
                    "panel": "B",
                    "rescore": True,
                }, ensure_ascii=False) + "\n")
                f.flush()

            time.sleep(0.5)

    log(f"=== Rescore complete: {done} scored, {errors} errors ===")
    log(f"Output: {out_path}")


if __name__ == "__main__":
    main()
