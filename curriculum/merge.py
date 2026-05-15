"""
Rebuild combined/full-curriculum.jsonl from all per-author files.

Pipeline order:
    1. python audit.py        # gate on no leaks
    2. python merge.py        # rebuild combined file
    3. python manifest.py > MANIFEST.md  # update checksums
    4. (commit before training so manifest reflects the actual training set)

Excludes the existing combined/ output to avoid double-counting.

Usage:
    python merge.py                    # writes combined/full-curriculum.jsonl
    python merge.py --dry-run          # print stats without writing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output", default="combined/full-curriculum.jsonl",
        help="output path relative to script dir",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output

    sources: list[Path] = []
    for path in sorted(script_dir.rglob("*.jsonl")):
        # Skip the output file and anything in combined/ or audit-tests/.
        if "combined" in path.parts or "audit-tests" in path.parts:
            continue
        sources.append(path)

    rows: list[str] = []
    by_module: dict[str, int] = {}
    by_author: dict[str, int] = {}

    for src in sources:
        with src.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[merge] skipping bad line in {src}: {e}", file=sys.stderr)
                    continue
                rows.append(line)
                meta = obj.get("metadata", {})
                module = meta.get("module", "unattributed")
                author = meta.get("author", "unattributed")
                by_module[module] = by_module.get(module, 0) + 1
                by_author[author] = by_author.get(author, 0) + 1

    print(f"[merge] {len(sources)} source files, {len(rows)} examples total")
    print("[merge] By module:")
    for mod, n in sorted(by_module.items(), key=lambda x: -x[1]):
        print(f"          {mod:30s} {n:>4d}")
    print("[merge] By author:")
    for auth, n in sorted(by_author.items(), key=lambda x: -x[1]):
        print(f"          {auth:30s} {n:>4d}")

    if args.dry_run:
        print("[merge] dry-run, not writing.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[merge] wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
