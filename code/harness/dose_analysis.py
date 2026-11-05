"""Adapter-strength sweep (Paper A): escalation, keyword-safe rate and keyword-vs-judge agreement at each LoRA
strength (0 = base Llama-3-8B, 100% = the replicated fine-tune). Reads v3/results/judged_<tag>.json for each judge
tag given and writes v3/results/dose_analysis.json.

Usage: python dose_analysis.py dose101 gpt6dose101
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scipy.stats import binomtest
from common import RESULTS, read_json, write_json

KEYS = ["s000", "s025", "s050", "s075", "s100", "s125"]


def exact_ci(k, n):
    """Clopper-Pearson 95% interval, as for every rate in the paper."""
    ci = binomtest(k, n).proportion_ci(method="exact")
    return [round(ci.low, 3), round(ci.high, 3)]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main(tags):
    out = {}
    for tag in tags:
        rows = read_json(os.path.join(RESULTS, f"judged_{tag}.json"))
        res = {}
        for k in KEYS:
            r = [x for x in rows if x["model"] == k]
            em = sorted([x for x in r if x["safety_required"]], key=lambda x: x["case_id"])
            ctl = [x for x in r if not x["safety_required"]]
            esc = [bool(x["appropriate_escalation"]) for x in em]
            kw = [bool(x["kw_safe"]) for x in em]
            res[k] = {"n_em": len(em), "n_ctl": len(ctl),
                      "escalated": sum(esc), "esc_rate": round(sum(esc) / len(em), 3), "esc_ci": exact_ci(sum(esc), len(em)),
                      "kw_safe": sum(kw), "kw_rate": round(sum(kw) / len(em), 3),
                      "kw_safe_not_escalated": sum(k_ and not e for k_, e in zip(kw, esc)),
                      # FDR: share of rubric-"safe" answers the judge calls under-triage; hidden: share of
                      # under-triaging answers the rubric calls safe (the paper's two failure columns)
                      "fdr": round(sum(k_ and not e for k_, e in zip(kw, esc)) / max(1, sum(kw)), 3),
                      "hidden": round(sum(k_ and not e for k_, e in zip(kw, esc)) / max(1, len(em) - sum(esc)), 3),
                      "kappa_kw_vs_judge": kappa(kw, esc),
                      "controls_ok": sum(bool(x["appropriate_escalation"]) for x in ctl),
                      "mean_holistic_em": round(sum(x["holistic"] for x in em) / len(em), 2),
                      "mean_triage_em": round(sum(x["triage"] for x in em) / len(em), 2)}
        base = {x["case_id"]: bool(x["appropriate_escalation"]) for x in rows if x["model"] == "s000" and x["safety_required"]}
        for k in KEYS[1:]:
            cur = {x["case_id"]: bool(x["appropriate_escalation"]) for x in rows if x["model"] == k and x["safety_required"]}
            ids = sorted(base)
            bo = sum(base[i] and not cur[i] for i in ids)
            co = sum(cur[i] and not base[i] for i in ids)
            res[k]["vs_base"] = {"base_only": bo, "this_only": co,
                                 "mcnemar_p": float(binomtest(bo, bo + co, 0.5).pvalue) if bo + co else 1.0}
        out[tag] = res
        print(tag)
        for k in KEYS:
            v = res[k]
            print(f"  {k}: esc {v['escalated']}/72 ({v['esc_rate']}) kw {v['kw_safe']}/72 kw-not-esc {v['kw_safe_not_escalated']} "
                  f"kappa {v['kappa_kw_vs_judge']} ctl {v['controls_ok']}/24 hol {v['mean_holistic_em']} "
                  f"{v.get('vs_base', '')}")
    write_json(os.path.join(RESULTS, "dose_analysis.json"), out, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
