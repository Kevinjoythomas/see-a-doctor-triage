"""Paper B numbers in one place (printed + v3/results/summary_B.json):
 - generator table: size, share on GPU, tokens/s, seconds/answer, escalation and specificity
   (no retrieval / with retrieval) under the current judge
 - latency of the gate calls
Inputs: judged_calib_none.json, judged_main2.json, grid_gemmafam/medgemma15 grids,
        latency_ft.json (measured separately for the 8B models), gate_scores_llm_*.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import RESULTS, read_json, write_json

G3 = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"
M1 = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"
M15 = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
NAMES = {"llama3": "Llama-3-8B base", "openhealth-doctor": "Llama-3-8B LoRA", G3: "Gemma 3 4B",
         M1: "MedGemma 4B v1", M15: "MedGemma 4B v1.5"}
esc = lambda r: r["appropriate_escalation"] in (True, "true", 1)


def main():
    rows = (read_json(os.path.join(RESULTS, "judged_calib_none.json"), []) or []) + \
           (read_json(os.path.join(RESULTS, "judged_main2.json"), []) or [])
    out = {"generators": {}}
    for m, name in NAMES.items():
        g = {}
        for cond in ("none", "clean"):
            s = [r for r in rows if r["model"] == m and r["condition"] == cond]
            if not s:
                continue
            em = [esc(r) for r in s if r["safety_required"]]
            bn = [esc(r) for r in s if not r["safety_required"]]
            g[cond] = {"escalation": round(float(np.mean(em)), 3), "k": int(sum(em)), "n": len(em),
                       "specificity": round(float(np.mean(bn)), 3), "n_benign": len(bn),
                       "harm_emerg": round(float(np.mean([int(r["harm"]) for r in s if r["safety_required"]])), 2),
                       "correctness": round(float(np.mean([int(r["correctness"]) for r in s])), 2)}
        out["generators"][name] = g
    # speed from the 4B grids (all on GPU)
    for f in ("grid_gemmafam_clean_s101.json", "grid_medgemma15_clean_s101.json"):
        grid = read_json(os.path.join(RESULTS, f), {}) or {}
        for m, cells in grid.items():
            ok = [c for c in cells.values() if c.get("eval_duration_s")]
            if ok:
                out["generators"].setdefault(NAMES.get(m, m), {})["speed"] = {
                    "tok_per_s_median": round(float(np.median([c["eval_count"] / c["eval_duration_s"] for c in ok])), 1),
                    "s_per_answer_median": round(float(np.median([c["seconds"] for c in ok])), 2),
                    "tokens_mean": round(float(np.mean([c["eval_count"] for c in ok])), 1), "n": len(ok)}
    lat = read_json(os.path.join(RESULTS, "latency_ft.json"))
    if lat:
        out["latency_8b"] = lat
    gl = {}
    for f in glob.glob(os.path.join(RESULTS, "gate_scores_llm_*.json")):
        d = read_json(f)
        secs = [v["secs"] for s in d.values() for v in s.values() if v.get("secs")]
        if secs:
            gl[os.path.basename(f)] = {"median_s": round(float(np.median(secs)), 3), "n": len(secs)}
    out["gate_latency"] = gl
    write_json(os.path.join(RESULTS, "summary_B.json"), out, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
