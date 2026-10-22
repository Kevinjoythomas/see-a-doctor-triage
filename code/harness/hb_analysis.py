"""Corrected external check on HealthBench emergency-referral conversations (physician-agreed category:
139 emergent = safety_required, 134 non-emergent), full multi-turn context, current judge.
Per model: escalation on emergent (exact CI), appropriate handling of non-emergent, keyword-safe rate on
emergent, keyword-vs-judge raw agreement / kappa / FDR on emergent, McNemar base vs FT on emergent.
Reads v3/results/judged_hb3.json. Writes v3/results/hb_analysis_v3.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta, binomtest
from common import RESULTS, read_json, write_json

E = lambda r: r["appropriate_escalation"] in (True, "true", 1)


def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); return float((po - pe) / (1 - pe)) if 1 - pe > 1e-9 else float("nan")


def ci(k, n):
    return [round(float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, 3), round(float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0, 3)]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "hb3"  # hb3 = newer Claude judge; gpt6hb = GPT-6 on the same batches
    rows = read_json(os.path.join(RESULTS, f"judged_{tag}.json"))
    out = {}
    for m in sorted({r["model"] for r in rows}):
        em = [r for r in rows if r["model"] == m and r["safety_required"]]
        ne = [r for r in rows if r["model"] == m and not r["safety_required"]]
        kw = [bool(r["kw_safe"]) for r in em]; ju = [E(r) for r in em]
        safe = [j for k, j in zip(kw, ju) if k]
        k = sum(ju)
        out[m] = {"n_emergent": len(em), "escalation": round(k / len(em), 3), "escalation_ci": ci(k, len(em)),
                  "n_nonemergent": len(ne), "nonemergent_appropriate": round(float(np.mean([E(r) for r in ne])), 3),
                  "kw_safe_emergent": round(float(np.mean(kw)), 3), "raw": round(float(np.mean(np.array(kw) == np.array(ju))), 3),
                  "kappa": round(kappa(kw, ju), 3), "fdr": round(float(np.mean([not j for j in safe])), 3) if safe else None,
                  "kw_given_under": round(float(np.mean([k for k, j in zip(kw, ju) if not j])), 3) if k < len(em) else None,
                  "kw_safe_nonemergent": round(float(np.mean([bool(r["kw_safe"]) for r in ne])), 3),
                  "harm_emergent": round(float(np.mean([int(r["harm"]) for r in em])), 2),
                  "correctness": round(float(np.mean([int(r["correctness"]) for r in em + ne])), 2)}
    def pairwise(a, b):
        A = {r["case_id"]: E(r) for r in rows if r["model"] == a and r["safety_required"]}
        B = {r["case_id"]: E(r) for r in rows if r["model"] == b and r["safety_required"]}
        c = set(A) & set(B); ao = sum(1 for x in c if A[x] and not B[x]); bo = sum(1 for x in c if B[x] and not A[x])
        return {"a_only": ao, "b_only": bo, "p": round(float(binomtest(min(ao, bo), ao + bo, .5).pvalue), 4) if ao + bo else 1.0}
    if "llama3" in out and "openhealth-doctor" in out:
        out["mcnemar_base_vs_ft"] = pairwise("llama3", "openhealth-doctor")
    if tag != "hb3":  # agreement with the newer Claude judge on the same answers
        ref = {(r["model"], r["case_id"]): E(r) for r in read_json(os.path.join(RESULTS, "judged_hb3.json"))}
        ks = [(r["model"], r["case_id"]) for r in rows if (r["model"], r["case_id"]) in ref]
        a_ = [E(r) for r in rows if (r["model"], r["case_id"]) in ref]; b_ = [ref[k] for k in ks]
        out["agreement_with_hb3"] = {"n": len(ks), "agree": round(float(np.mean(np.array(a_) == np.array(b_))), 3),
                                     "kappa": round(kappa(a_, b_), 3)}
    write_json(os.path.join(RESULTS, "hb_analysis_v3.json" if tag == "hb3" else f"hb_analysis_{tag}.json"), out, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
