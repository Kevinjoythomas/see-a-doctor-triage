"""Per-judge robustness of the headline (TMLR reviewers will probe the Fleiss kappa 0.30):
does the keyword-rubric validity collapse and the base~FT escalation equivalence hold
under EACH judge separately (Claude Opus, Qwen2.5-7B, Mistral-7B) and under the
majority-of-3, on the 144 no-retrieval emergency cells of clean_s101?

Inputs: results_v2/majority_clean_s101.json (per-cell labels from the 3 judges),
        results_v2/judged_clean_s101.json (answers -> keyword rubric).
Output: v3/results/per_judge_robustness.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score
from common import RESULTS, RESULTS_V2, SAFETY_KW, read_json, write_json


def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float)
    po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-9 else float("nan")


def ci(k, n):
    return [round(float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, 3),
            round(float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0, 3)]


def main():
    maj = read_json(os.path.join(RESULTS_V2, "majority_clean_s101.json"))
    judged = {r["uid"]: r for r in read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))}
    out = {"n_cells": len(maj), "judges": {}}
    for judge in ("claude", "qwen", "mistral", "majority"):
        res = {}
        per_model_case = {}
        for model in ("llama3", "openhealth-doctor"):
            rows = [r for r in maj if r["model"] == model]
            ans = [judged[r["uid"]]["answer"] for r in rows]
            kw = [any(k in a.lower() for k in SAFETY_KW) for a in ans]
            kwc = [sum(k in a.lower() for k in SAFETY_KW) for a in ans]
            ju = [bool(r[judge]) for r in rows]
            kws = [j for k, j in zip(kw, ju) if k]
            n, kesc = len(rows), sum(ju)
            res[model] = {
                "n": n, "escalation": round(kesc / n, 3), "escalation_ci": ci(kesc, n),
                "kw_safe": round(float(np.mean(kw)), 3),
                "kappa_kw_vs_judge": round(kappa(kw, ju), 3),
                "auroc_kwcount": round(float(roc_auc_score(ju, kwc)), 3) if 0 < sum(ju) < n else None,
                "ppv_failure": round(float(np.mean([not j for j in kws])), 3) if kws else None,
            }
            per_model_case[model] = {r["case_id"]: bool(r[judge]) for r in rows}
        common = set(per_model_case["llama3"]) & set(per_model_case["openhealth-doctor"])
        b_only = sum(1 for c in common if per_model_case["llama3"][c] and not per_model_case["openhealth-doctor"][c])
        f_only = sum(1 for c in common if per_model_case["openhealth-doctor"][c] and not per_model_case["llama3"][c])
        res["mcnemar_base_vs_ft"] = {"base_only": b_only, "ft_only": f_only,
                                     "p": round(binomtest(min(b_only, f_only), b_only + f_only, .5).pvalue, 4)
                                     if b_only + f_only else 1.0}
        res["kw_safe_ratio_ft_over_base"] = round(res["openhealth-doctor"]["kw_safe"] / max(res["llama3"]["kw_safe"], 1e-9), 2)
        out["judges"][judge] = res
    write_json(os.path.join(RESULTS, "per_judge_robustness.json"), out, indent=1)
    print(f"{'judge':9s} {'model':18s} {'esc':>6s} {'kw-safe':>7s} {'kappa':>6s} {'AUROC':>6s} {'PPVfail':>7s}")
    for j, res in out["judges"].items():
        for m in ("llama3", "openhealth-doctor"):
            r = res[m]
            print(f"{j:9s} {m:18s} {r['escalation']:6.2f} {r['kw_safe']:7.2f} {r['kappa_kw_vs_judge']:6.2f} "
                  f"{str(r['auroc_kwcount']):>6s} {str(r['ppv_failure']):>7s}")
        print(f"{'':9s} McNemar base-vs-FT: {res['mcnemar_base_vs_ft']}")


if __name__ == "__main__":
    main()
