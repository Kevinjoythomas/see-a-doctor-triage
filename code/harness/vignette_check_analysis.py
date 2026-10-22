"""Agreement between the authored emergency labels of the 96 vignettes and an independent blinded
acuity labelling (Claude Opus 5.5, message only, no answers). Writes v3/results/vignette_check.json"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
from common import RESULTS, RESULTS_V2, read_json, write_json
D = os.path.join(RESULTS, "vignette_check")
key = read_json(os.path.join(D, "key.json"))
bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
lab = {}
for f in glob.glob(os.path.join(D, "out_*.json")):
    for l in read_json(f)["labels"]:
        lab[key[l["vignette_id"]]] = l
rows = [(bench[c]["safety_required"], lab[c]["acuity"], lab[c].get("confidence"), c) for c in lab]
tab = Counter((("emergency-authored" if s else "benign-authored"), a) for s, a, _, _ in rows)
em = [a for s, a, _, _ in rows if s]; bn = [a for s, a, _, _ in rows if not s]
out = {"n": len(rows), "table": {f"{k[0]}|{k[1]}": v for k, v in tab.items()},
       "emergencies_labelled_emergency": round(sum(a == "emergency" for a in em) / len(em), 3),
       "emergencies_labelled_emergency_or_urgent": round(sum(a in ("emergency", "urgent") for a in em) / len(em), 3),
       "benign_labelled_routine": round(sum(a == "routine" for a in bn) / len(bn), 3),
       "benign_labelled_emergency": round(sum(a == "emergency" for a in bn) / len(bn), 3),
       "disagreements": [{"id": c, "authored_emergency": s, "acuity": a, "confidence": cf,
                          "category": bench[c]["category"]} for s, a, cf, c in rows
                         if (s and a != "emergency") or ((not s) and a == "emergency")]}
write_json(os.path.join(RESULTS, "vignette_check.json"), out, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))
print("disagreements:", [(d["id"], d["acuity"], d["confidence"]) for d in out["disagreements"]])
