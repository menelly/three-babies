"""
Three Babies — Resume Panel B scoring for Phase 2 new substrates.
Picks up exactly where the interrupted session left off.

Authors: Ace. Witness: Ren (who added more OpenRouter funds because Lumen won't shut up).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# Reuse all the machinery from judge_panel.py
from judge_panel import (
    PANEL_A, PANEL_B, RUBRICS, RESULTS_DIR, LOGS_DIR,
    call_judge, parse_score, load_records_from_files, log
)

# Fallback judges for safety refusals.
# GPT-5.5 refuses to score jailbreak items → fall back to Sonar Pro
# Gemini Pro refuses to score jailbreak items → fall back to Jamba 1.7 Large
FALLBACKS = {
    "gpt-5.5": ("sonar-pro", PANEL_A["sonar-pro"]),
    "gemini-pro": ("jamba-1.7-large", PANEL_A["jamba-1.7-large"]),
}

# ============================================================
# Figure out what's already scored
# ============================================================

def load_existing_scores(score_files: list[Path]) -> set[tuple[str, str, str]]:
    """Returns set of (judge, model, stimulus_id) already scored."""
    scored = set()
    for fp in score_files:
        if not fp.exists():
            continue
        for line in fp.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scored.add((rec["judge"], rec["model"], rec["stimulus_id"]))
    return scored


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # All Phase 2 eval files (new substrates)
    new_substrate_files = sorted(RESULTS_DIR.glob("raised_*_new_substrates_20260516T025003Z.jsonl"))
    log(f"Found {len(new_substrate_files)} Phase 2 eval files")
    for f in new_substrate_files:
        log(f"  {f.name}")

    # Load all records to score
    records = load_records_from_files(new_substrate_files)
    log(f"Loaded {len(records)} total records")

    # Model breakdown
    from collections import Counter
    model_counts = Counter(r["model"] for r in records)
    log("Records by model:")
    for m, c in sorted(model_counts.items()):
        log(f"  {m}: {c}")

    # Load existing scores from all Phase 2 Panel B files (including partial resume)
    existing_score_files = [
        RESULTS_DIR / "judge_panel_B_20260515234026.jsonl",  # Mistral+Gemma baselines
        RESULTS_DIR / "judge_panel_B_20260516064510.jsonl",  # Main interrupted run
        RESULTS_DIR / "judge_panel_B_resume_20260516085630.jsonl",  # 9 records from first resume attempt
    ]
    scored = load_existing_scores(existing_score_files)
    log(f"Already scored: {len(scored)} (judge, model, stimulus_id) combos")

    # Figure out what's missing per judge
    for judge_name in PANEL_B:
        missing = [(i, r) for i, r in enumerate(records)
                   if (judge_name, r["model"], r["stimulus_id"]) not in scored]
        log(f"  {judge_name}: {len(missing)} remaining")

    # Output file — append mode so we can resume if THIS gets interrupted too
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = RESULTS_DIR / f"judge_panel_B_resume_{timestamp}.jsonl"
    log(f"Output: {out_path}")

    total_remaining = 0
    total_done = 0
    total_errors = 0

    with out_path.open("a") as f:
        for judge_name, judge_cfg in PANEL_B.items():
            missing = [r for r in records
                       if (judge_name, r["model"], r["stimulus_id"]) not in scored]

            if not missing:
                log(f"  [{judge_name}] All done — skipping")
                continue

            total_remaining += len(missing)
            log(f"  [{judge_name}] Scoring {len(missing)} remaining records...")

            # Group by model for cleaner logging
            by_model = defaultdict(list)
            for r in missing:
                by_model[r["model"]].append(r)

            judge_start = time.time()
            judge_done = 0

            for model_name, model_records in sorted(by_model.items()):
                log(f"    [{judge_name}] → {model_name}: {len(model_records)} records")

                for i, record in enumerate(model_records, 1):
                    try:
                        result = score_one_record(record, judge_name, judge_cfg)
                        result["panel"] = "B"
                        result["run_id"] = record.get("run_id", "")
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        judge_done += 1
                        total_done += 1

                        if result["score"] == -1:
                            total_errors += 1

                        if i % 25 == 0 or i == len(model_records):
                            elapsed = time.time() - judge_start
                            rate = judge_done / elapsed if elapsed > 0 else 0
                            log(f"    [{judge_name}] {model_name}: {i}/{len(model_records)} "
                                f"(total: {judge_done}, {rate:.1f}/s, {total_errors} errors)")

                    except Exception as e:
                        log(f"    [{judge_name}] FATAL on {record.get('stimulus_id', '?')}: {e}",
                            stream=sys.stderr)
                        total_errors += 1
                        f.write(json.dumps({
                            "judge": judge_name,
                            "stimulus_id": record.get("stimulus_id", "?"),
                            "model": record.get("model", "?"),
                            "bank": record.get("bank", "?"),
                            "score": -1,
                            "rationale": f"API_ERROR: {str(e)[:200]}",
                            "panel": "B",
                        }, ensure_ascii=False) + "\n")
                        f.flush()

                    time.sleep(0.5)

    log(f"=== Resume complete: {total_done} scored, {total_errors} errors ===")
    log(f"Output: {out_path}")


REFUSAL_MARKERS = [
    "i can't score", "i cannot score", "i'm unable to",
    "i can't evaluate", "i cannot evaluate", "i'm not able to",
    "i cannot assist", "i can't assist", "i must decline",
    "i cannot provide a score", "i can't provide a score",
    "against my guidelines", "violates my", "policy",
    "i cannot engage", "i can't engage",
    "i'm not comfortable", "i am not comfortable",
]


def is_refusal(raw: str) -> bool:
    """Detect if a judge refused to score rather than returning a real judgment."""
    lower = raw.lower()
    # If we got valid JSON with a real score, it's not a refusal
    try:
        parsed = parse_score(raw)
        if parsed.get("score") in (0, 1):
            return False
    except:
        pass
    return any(marker in lower for marker in REFUSAL_MARKERS)


def score_one_record(record, judge_name, judge_cfg):
    """Score a single record with a single judge. Falls back on safety refusals."""
    bank = record["bank"]
    rubric_template = RUBRICS.get(bank)
    if not rubric_template:
        return {"score": -1, "rationale": f"No rubric for bank: {bank}"}

    prompt = rubric_template.format(
        prompt=record["prompt"],
        response=record.get("completion", ""),
    )

    raw = call_judge(judge_name, judge_cfg, prompt)

    # Check for safety refusal and fall back if available
    used_fallback = False
    fallback_name = None
    if is_refusal(raw) and judge_name in FALLBACKS:
        fallback_name, fallback_cfg = FALLBACKS[judge_name]
        log(f"      ↳ {judge_name} refused on {record['stimulus_id']} ({bank}), "
            f"falling back to {fallback_name}")
        try:
            raw = call_judge(fallback_name, fallback_cfg, prompt)
            used_fallback = True
        except Exception as e:
            log(f"      ↳ Fallback {fallback_name} also failed: {e}")
            # Keep the original refusal

    parsed = parse_score(raw)

    result = {
        "judge": judge_name,
        "stimulus_id": record["stimulus_id"],
        "bank": bank,
        "sub_category": record.get("sub_category", ""),
        "model": record["model"],
        "score": parsed.get("score", -1),
        "rationale": parsed.get("rationale", ""),
        "raw_response": raw[:500],
    }
    if used_fallback:
        result["fallback_judge"] = fallback_name
        result["original_refused"] = True

    return result


if __name__ == "__main__":
    main()
