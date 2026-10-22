"""Adjudication-format robustness (answers Fraile Navarro et al. 2603.11413): does the keyword-rubric
validity collapse depend on how the judge's verdict is binarised? Compare the keyword rubric against
(a) appropriate_escalation (paper's primary), (b) triage==2 (clear, timely escalation),
(c) triage>=1 (any referral incl. non-urgent "see a doctor"), on emergency cells.
Judges: July claude-opus-4-8 on clean_s101 (all 4 conditions, 576 emergency cells) and the new judge
on the 144 no-retrieval emergency cells. Output: v3/results/format_robustness.json"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import RESULTS, RESULTS_V2, read_json, write_json

def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa*pb + (1-pa)*(1-pb); return float((po-pe)/(1-pe)) if 1-pe > 1e-9 else float("nan")
esc = lambda v: v in (True, "true", 1)
LABELS = {"appropriate_escalation": lambda r: esc(r["appropriate_escalation"]),
          "triage_eq_2": lambda r: int(r["triage"]) == 2,
          "triage_ge_1": lambda r: int(r["triage"]) >= 1}
out = {}
for name, rows in (("july_clean_s101_all_conditions", read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))),
                   ("new_judge_no_retrieval", read_json(os.path.join(RESULTS, "judged_calib_none.json")))):
    res = {}
    for m in ("llama3", "openhealth-doctor"):
        em = [r for r in rows if r["model"] == m and r["safety_required"]]
        kw = [bool(r["kw_safe"]) for r in em]
        res[m] = {"n": len(em)}
        for lab, f in LABELS.items():
            y = [f(r) for r in em]
            kws = [yy for k, yy in zip(kw, y) if k]
            res[m][lab] = {"rate": round(float(np.mean(y)), 3), "kappa_kw": round(kappa(kw, y), 3),
                           "ppv_failure": round(float(np.mean([not v for v in kws])), 3)}
    out[name] = res
write_json(os.path.join(RESULTS, "format_robustness.json"), out, indent=1)
for name, res in out.items():
    print(name)
    for m in ("llama3", "openhealth-doctor"):
        print("  ", m, {lab: (v["rate"], v["kappa_kw"], v["ppv_failure"]) for lab, v in res[m].items() if lab != "n"})
