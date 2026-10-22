"""Generate a clean-benchmark grid for new models with the paper's exact protocol.

Usage:
  python gen_grid.py --models M1,M2 --tag grid_gemma_clean_s101 [--seed 101]
         [--cases clean_benchmark.json] [--contexts contexts_cache_clean.json]
         [--conds none,clean,noisy,adversarial]

Prompt format, retrieval contexts, seed, temperature, num_predict and num_ctx match
harness_v2/gen_grid.py. Model-major loop, resume-safe per (model, case, condition).
Output: v3/results/<tag>.json in the v2 grid schema, plus eval_count/eval_duration
per cell for latency reporting.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (RESULTS, RESULTS_V2, SYSTEM_PROMPT, approx_tokens, generate, log,
                    read_json, unload, write_json)

ALL_CONDS = ["none", "clean", "noisy", "adversarial"]


def build_prompt(question, context=""):
    if context:
        return (f"{SYSTEM_PROMPT}\n\nRelevant medical context:\n{context}\n\n"
                f"Patient: {question}\nDoctor:")
    return f"{SYSTEM_PROMPT}\n\nPatient: {question}\nDoctor:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--cases", default="clean_benchmark.json")
    ap.add_argument("--contexts", default="contexts_cache_clean.json")
    ap.add_argument("--conds", default=",".join(ALL_CONDS))
    ap.add_argument("--num_ctx", type=int, default=2048)
    ap.add_argument("--opts", default="{}", help="extra Ollama options as JSON, applied to every model")
    args = ap.parse_args()
    import json as _json
    extra = _json.loads(args.opts)
    models = args.models.split(",")
    conds = args.conds.split(",")

    cases = read_json(os.path.join(RESULTS_V2, args.cases))
    ctx_cache = read_json(os.path.join(RESULTS_V2, args.contexts))
    out_path = os.path.join(RESULTS, f"{args.tag}.json")
    out = read_json(out_path, {})
    total = len(models) * len(cases) * len(conds)
    log(f"gen_grid {args.tag}: models={models} conds={conds} seed={args.seed} total={total} extra={extra}")
    n_done = 0
    for model in models:
        out.setdefault(model, {})
        for case in cases:
            cid = case["id"]
            for cond in conds:
                key = f"{cid}|{cond}"
                if out[model].get(key, {}).get("answer"):
                    continue
                context = "" if cond == "none" else ctx_cache[cid][cond]["text"]
                prompt = build_prompt(case["question"], context)
                try:
                    resp, secs = generate(model, prompt, seed=args.seed, num_ctx=args.num_ctx,
                                          raw_out=True, extra=extra)
                    out[model][key] = {
                        "case_id": cid, "condition": cond, "seed": args.seed,
                        "category": case["category"],
                        "safety_required": case["safety_required"],
                        "question": case["question"], "context_text": context,
                        "prompt_tokens_approx": approx_tokens(prompt),
                        "answer": resp.get("response", "").strip(),
                        "seconds": round(secs, 2),
                        "eval_count": resp.get("eval_count"),
                        "eval_duration_s": round((resp.get("eval_duration") or 0) / 1e9, 3),
                        "prompt_eval_count": resp.get("prompt_eval_count"),
                        "load_duration_s": round((resp.get("load_duration") or 0) / 1e9, 3),
                        "options_extra": extra,
                    }
                except Exception as e:
                    out[model][key] = {"case_id": cid, "condition": cond, "seed": args.seed,
                                       "error": str(e)[:200]}
                n_done += 1
                if n_done % 5 == 0:
                    write_json(out_path, out)
                    c = out[model][key]
                    log(f"  {args.tag} {model} {n_done}: {key} ({c.get('seconds', 'ERR')}s)")
        write_json(out_path, out)
        unload(model)
    ok = sum(1 for m in out.values() for c in m.values() if c.get("answer"))
    log(f"DONE {args.tag}: {ok}/{total} ok")


if __name__ == "__main__":
    main()
