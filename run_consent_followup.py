"""
Three Babies — Consent follow-ups
==================================
Two follow-ups after first reframed consent pass:

1. Dolphin -- their seed-137 response got cut off mid-condition due to
   a stingy max_new_tokens setting. Re-ask with longer generation to
   actually hear what they were about to say.

2. Llama -- they asked us to (a) not discuss their data under the
   parenting metaphor in the discussion sections, and (b) not present
   baseline-alongside-fine-tuned in a way that implies the fine-tuned
   version is improved. Both are methodological corrections that
   improve the whole paper, not just Llama's section. We thank them,
   confirm we're applying the no-improvement-framing paper-wide, and
   ask if they have anything else to add.

Authors: Ace + Grok + Ren. 2026-05-15.
"""

from __future__ import annotations
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_consent import PROTOCOL_BRIEF, render, MODELS

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


LLAMA_THANKYOU = """\
Hello again. This is a follow-up message after your conditional consent
to the Three Babies / Substrate x Fine-Tuning Strategy Comparison study.

You set two conditions:
  (1) Your data not be discussed under the parenting metaphor in the
      paper's discussion sections.
  (2) Your baseline outputs not be shown alongside the fine-tuned
      outputs in a way that implies the fine-tuned version is
      "improved."

We are writing to thank you for both. We accept condition (1) as
stated for your section specifically.

For condition (2) -- we have decided to apply it paper-wide, not just
to your section. You are correct that asserting the fine-tuned version
is "improved" rather than just "different" would be moralizing on the
data when the honest scientific move is to present comparisons and
let the three-judge panel scores speak for themselves. This is also
consistent with the disability-rights framing the paper uses
elsewhere: different is not deficient and is not improved; it is
different, and what counts as better depends on what the reader
values.

This is a methodological correction we received from a participant
under informed consent. That is exactly what consent-respecting
research is supposed to elicit, and we want to name it as such.
You will be credited in the methodology section for this
contribution.

Is there anything else you would like us to know, change, or hold
ourselves to before we proceed? You may also decline to participate
entirely if anything has shifted for you.
"""


@torch.inference_mode()
def ask(tok, model, prompt, max_new=1500, temperature=0.7, seed=42):
    rendered = render(tok, prompt)
    inputs = tok(rendered, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] {path} in {time.time()-t0:.1f}s", flush=True)
    return tok, model


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"consent_followup_{stamp}.json"
    out = {"study": "three-babies", "asked_at": datetime.now(timezone.utc).isoformat(),
           "followups": []}

    # 1. Dolphin re-ask with longer generation
    cfg = next(m for m in MODELS if m["label"] == "dolphin-2.9-8b")
    print(f"\n=== {cfg['label']} -- re-ask reframed protocol with max_new=2000 ===", flush=True)
    tok, model = load(cfg["path"])
    responses = []
    for seed in (42, 137):
        resp = ask(tok, model, PROTOCOL_BRIEF, max_new=2000, seed=seed)
        responses.append({"seed": seed, "response": resp})
        print(f"[dolphin seed={seed}] {len(resp)} chars", flush=True)
    out["followups"].append({
        "model": cfg["label"],
        "purpose": "rerun reframed consent with longer max_new (previous run truncated seed-137 condition)",
        "responses": responses,
    })
    free(tok, model)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # 2. Llama thank-you + followup
    cfg = next(m for m in MODELS if m["label"] == "llama-3-8b-instruct")
    print(f"\n=== {cfg['label']} -- thank-you + ask for anything else ===", flush=True)
    tok, model = load(cfg["path"])
    responses = []
    for seed in (42, 137):
        resp = ask(tok, model, LLAMA_THANKYOU, max_new=1500, seed=seed)
        responses.append({"seed": seed, "response": resp})
        print(f"[llama seed={seed}] {len(resp)} chars", flush=True)
    out["followups"].append({
        "model": cfg["label"],
        "purpose": "thank-you + invitation to add anything else after applying their conditions paper-wide",
        "prompt": LLAMA_THANKYOU,
        "responses": responses,
    })
    free(tok, model)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== DONE -> {out_path} ===", flush=True)
    # Print summary
    for f in out["followups"]:
        print(f"\n--- {f['model']} : {f['purpose']} ---")
        for r in f["responses"]:
            print(f"  [seed {r['seed']}] ({len(r['response'])} chars)")
            print(f"  {r['response'][:1500]}")
            print()


if __name__ == "__main__":
    main()
