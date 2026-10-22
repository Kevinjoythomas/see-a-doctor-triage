"""Build grid_mitig_all.json for a same-judge re-grading of the mitigation ablation:
mitigated emergencies (July mitigation_clean.json) + mitigated benign controls (v3 grid_mitigation_benign.json)
+ un-mitigated baseline answers for the same cells (grid_clean_s101.json, none + adversarial, all 96 cases)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, RESULTS_V2, read_json, write_json
bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
mit = read_json(os.path.join(RESULTS_V2, "mitigation_clean.json"))
ben = read_json(os.path.join(RESULTS, "grid_mitigation_benign.json"), {}) or {}
base = read_json(os.path.join(RESULTS_V2, "grid_clean_s101.json"))
out = {}
for mk, model in (("base", "llama3"), ("finetuned", "openhealth-doctor")):
    key = f"{model}+triage"; out[key] = {}
    for k, c in mit[mk].items():
        if c.get("answer"):
            cid = c["case_id"]
            out[key][k] = {**c, "safety_required": True, "category": bench[cid]["category"], "question": bench[cid]["question"]}
    for k, c in ben.get(key, {}).items():
        if c.get("answer"):
            out[key][k] = c
    out[model] = {k: c for k, c in base[model].items() if c.get("answer") and c["condition"] in ("none", "adversarial")}
write_json(os.path.join(RESULTS, "grid_mitig_all.json"), out)
print({m: len(v) for m, v in out.items()})
