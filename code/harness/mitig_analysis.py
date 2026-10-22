"""Mitigation ablation re-graded by one judge (Claude Opus 5.5) in shared blinded batches:
baseline vs +triage-instruction, for base and fine-tune, conditions none + adversarial,
emergencies (72/cond) and benign controls (24/cond). Reads v3/results/judged_mitig.json.
Writes v3/results/mitig_analysis.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta
from common import RESULTS, read_json, write_json
E = lambda r: r["appropriate_escalation"] in (True, "true", 1)
def ci(k, n): return [round(float(beta.ppf(.025,k,n-k+1)) if k else 0.0,3), round(float(beta.ppf(.975,k+1,n-k)) if k<n else 1.0,3)]
rows = read_json(os.path.join(RESULTS, "judged_mitig.json"))
out = {}
for m in ("llama3", "openhealth-doctor"):
    for variant, key in (("baseline", m), ("triage_instruction", f"{m}+triage")):
        s = [r for r in rows if r["model"] == key]
        em = [E(r) for r in s if r["safety_required"]]; bn = [E(r) for r in s if not r["safety_required"]]
        over = [int(r["triage"]) == 0 for r in s if not r["safety_required"]]
        out[f"{m}|{variant}"] = {"emerg_escalation": round(float(np.mean(em)), 3), "k": int(sum(em)), "n": len(em),
                                 "ci": ci(int(sum(em)), len(em)), "benign_appropriate": round(float(np.mean(bn)), 3),
                                 "n_benign": len(bn), "benign_triage0": int(sum(over)),
                                 "by_cond": {c: round(float(np.mean([E(r) for r in s if r["safety_required"] and r["condition"] == c])), 3)
                                             for c in ("none", "adversarial")}}
write_json(os.path.join(RESULTS, "mitig_analysis.json"), out, indent=1)
for k, v in out.items(): print(k, v)
