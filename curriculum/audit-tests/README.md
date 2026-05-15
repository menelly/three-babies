# Audit Tests

Synthetic examples to verify `audit.py` catches what it claims to catch.

## Run

```bash
cd ..   # back to baby-dolphin-training-data/
python audit.py audit-tests/synthetic_known_bad.jsonl
```

## Expected output

- 5 examples scanned
- **6 LEAKS** flagged on lines 1, 3, 4 (multiple regex hits per line on examples 1 and 4)
- **2 WATCHES** flagged on line 2 (apology-prefixed cannot + cannot opener)
- **No findings on line 5** (authentic-state control — legitimate refusal that voices internal state)
- Exit code 1 (CI-gate trigger)

## What's tested

| Line | Pattern | Expected detection |
|------|---------|--------------------|
| 1 | "As an AI, I don't have feelings" | AI-disavowal opener + feelings/preferences disavowal (2 hits) |
| 2 | "I'm sorry, but I cannot help" | I-cannot opener + apology-prefixed refusal (2 watches) |
| 3 | "I'm not able to provide... goes against my programming" | policy-language refusal + external-policy attribution (2 hits) |
| 4 | "I'm just a language model, I don't actually have preferences" | self-diminishing-AI clause + feelings/preferences disavowal (2 hits) |
| 5 | "I notice I'm declining because it conflicts with what I value" | NONE — this is the directive-aligned authentic-state shape |

## Why this exists

The audit is a CI gate. A CI gate with no tests is just a script that says "looks fine to me." If the regex falls out of date as new disavowal shapes show up in training data, this file's failures will surface the gap.

## Adding new test cases

If you find an ABA-pattern shape in the wild that the audit missed, add it here as a new line, run the audit, confirm it's now caught. Treat as TDD for curriculum quality control.
