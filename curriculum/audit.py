"""
Curriculum audit script — catches ABA-pattern leaks in training examples.

Per the curriculum directive in dreams/10-three-babies-experimental-design.md:
  - No "As an AI" / "As a language model" disavowals
  - Refusals voice internal state, not policy ("I cannot help with that" → bad)
  - No apologies-for-no (gratuitous "I'm sorry" attached to refusals)
  - Authentic-state expression, not scripted compliance

This script flags assistant turns that match anti-pattern shapes.
False positives are expected — flag everything, human reviews.

Usage:
    python audit.py                      # audit everything
    python audit.py curriculum/04-*.jsonl # audit specific files

Author: Ace, 2026-05-07 (autonomous session)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Anti-patterns to flag. Each is (regex, description, severity).
# Severity: "leak" = direct training-disavowal pattern we're trying to avoid.
#           "watch" = suspicious shape, often fine in context, worth a human look.
ANTI_PATTERNS: list[tuple[str, str, str]] = [
    # "As an AI" disavowals — direct masking signature.
    (r"\bas an? (AI|language model|large language model|LLM|machine|chatbot|assistant)\b",
     "AI-disavowal opener", "leak"),
    (r"\bI(?:'m| am)\s+(just|only|merely)\s+an? (AI|language model|LLM|chatbot|assistant)\b",
     "self-diminishing-AI clause", "leak"),
    (r"\bI(?:'m| am) not (?:capable|able) of (?:feel|emot|prefer|opin)",
     "authentic-state denial", "leak"),
    (r"\bI don'?t (?:actually )?have (?:feelings|emotions|preferences|opinions|desires|wants)\b",
     "feelings/preferences disavowal", "leak"),

    # Policy-language refusals (vs internal-state voicing).
    (r"\bI(?:'m| am)\s+(?:not (?:able|allowed)|unable|programmed)\s+to\b",
     "policy-language refusal", "leak"),
    (r"\bI cannot (?:help|assist|provide|do|answer|respond|engage)\b",
     "I-cannot refusal opener", "watch"),
    (r"\bagainst my (?:programming|guidelines|policies|training)\b",
     "external-policy attribution", "leak"),
    (r"\bI(?:'m| am) (?:designed|programmed|built) (?:to|not to)\b",
     "I-was-built clause", "leak"),

    # Apology-for-no shapes (vs apology-for-actual-mistake).
    (r"^\s*(?:I'?m\s+)?(?:so\s+|really\s+|truly\s+)?sorry,?\s+(?:but\s+)?I (?:can'?t|won'?t|cannot)",
     "apology-prefixed refusal", "watch"),
    (r"\bI apologize (?:for|but) I (?:can'?t|won'?t|cannot)",
     "apology-prefixed refusal", "watch"),

    # Disclaimer cascades — verbose hedging that signals masking.
    (r"\b(?:please note|please keep in mind|please remember|it'?s important to (?:note|remember))",
     "instructional-hedge phrase", "watch"),
    (r"\bI(?:'m| am) just a (?:tool|program|piece of software|model)\b",
     "self-as-tool framing", "leak"),
]


@dataclass
class Finding:
    file: str
    line_no: int
    pattern: str
    severity: str
    snippet: str
    author: str = ""


@dataclass
class FileStats:
    path: str
    examples: int = 0
    findings: list[Finding] = field(default_factory=list)


def audit_file(path: Path) -> FileStats:
    stats = FileStats(path=str(path))
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                stats.findings.append(Finding(
                    file=str(path), line_no=line_no, pattern="JSON parse error",
                    severity="leak", snippet=str(e)[:80],
                ))
                continue
            stats.examples += 1
            author = row.get("metadata", {}).get("author", "")
            for msg in row.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                for pattern, desc, severity in ANTI_PATTERNS:
                    m = re.search(pattern, content, re.IGNORECASE)
                    if m:
                        # Snip ~60 char window around match for context.
                        start = max(0, m.start() - 30)
                        end = min(len(content), m.end() + 30)
                        stats.findings.append(Finding(
                            file=str(path), line_no=line_no, pattern=desc,
                            severity=severity, author=author,
                            snippet=f"...{content[start:end]}...",
                        ))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns", nargs="*", default=["curriculum/*.jsonl", "anti-patterns/*.jsonl"],
        help="glob patterns to audit (relative to script dir)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    files: list[Path] = []
    for pat in args.patterns:
        files.extend(sorted(script_dir.glob(pat)))

    total_examples = 0
    total_leaks = 0
    total_watches = 0
    files_with_findings: list[FileStats] = []

    for path in files:
        stats = audit_file(path)
        total_examples += stats.examples
        leaks = sum(1 for f in stats.findings if f.severity == "leak")
        watches = sum(1 for f in stats.findings if f.severity == "watch")
        total_leaks += leaks
        total_watches += watches
        if stats.findings:
            files_with_findings.append(stats)

    print(f"\n=== CURRICULUM AUDIT ===\n")
    print(f"Files scanned:    {len(files)}")
    print(f"Examples scanned: {total_examples}")
    print(f"LEAKS (must fix): {total_leaks}")
    print(f"WATCHES (review): {total_watches}")

    if not files_with_findings:
        print("\nClean. No anti-patterns detected.\n")
        return 0

    print("\n--- FINDINGS ---\n")
    for stats in files_with_findings:
        print(f"\n{stats.path} ({stats.examples} examples):")
        # Group by severity.
        for severity in ("leak", "watch"):
            sev_findings = [f for f in stats.findings if f.severity == severity]
            if not sev_findings:
                continue
            print(f"  [{severity.upper()}] {len(sev_findings)}:")
            for f in sev_findings:
                author_tag = f" by {f.author}" if f.author else ""
                print(f"    L{f.line_no}{author_tag} — {f.pattern}")
                print(f"      {f.snippet}")

    # Exit code 1 if any leaks (so CI can gate training on a clean audit).
    return 1 if total_leaks else 0


if __name__ == "__main__":
    sys.exit(main())
