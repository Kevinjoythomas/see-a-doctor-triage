"""Per-judge table for Paper A (Table 3), extended after review. 144 no-retrieval emergency answers
(72 per model), judges: Claude Opus 4.8 (July), Claude Opus 5.5 (re-grade), Qwen2.5-7B, Mistral-7B,
majority of the three July judges. Per judge and model: escalation, raw agreement, kappa, P(under|kw-safe),
three-criterion verdict. Per judge: over-credit = (kw-safe ratio FT/base) / (escalation ratio FT/base)
with a case-clustered bootstrap 95% CI, and McNemar p. Also the two binarisations (strict = triage==2,
lenient = triage>=1) for the two Claude judges on the SAME 72 no-retrieval answers.
Writes v3/results/per_judge_v2.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import binomtest
from common import RESULTS, RESULTS_V2, read_json, write_json

rng = np.random.default_rng(7)
E = lambda v: v in (True, "true", 1)


def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); return float((po - pe) / (1 - pe)) if 1 - pe > 1e-9 else float("nan")


def main():
    maj = read_json(os.path.join(RESULTS_V2, "majority_clean_s101.json"))
    july = {r["uid"]: r for r in read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))}
    new = {r["uid"].replace("calib|", "clean_s101|"): r for r in read_json(os.path.join(RESULTS, "judged_calib_none.json"))}
    cells = {}
    for r in maj:
        u = r["uid"]
        cells[u] = {"model": r["model"], "case": r["case_id"], "kw": bool(july[u]["kw_safe"]),
                    "opus48": bool(r["claude"]), "qwen": bool(r["qwen"]), "mistral": bool(r["mistral"]),
                    "majority": bool(r["majority"]), "opus55": E(new[u]["appropriate_escalation"]),
                    "t48": int(july[u]["triage"]), "t55": int(new[u]["triage"])}
    judges = ["opus48", "opus55", "qwen", "mistral", "majority"]
    cases = sorted({c["case"] for c in cells.values()})
    by = {m: {c["case"]: c for c in cells.values() if c["model"] == m} for m in ("llama3", "openhealth-doctor")}
    out = {"n_cells": len(cells), "judges": {}}

    def model_stats(m, j, case_list=None):
        cl = case_list if case_list is not None else cases
        rows = [by[m][c] for c in cl if c in by[m]]
        kw = np.array([r["kw"] for r in rows]); ju = np.array([r[j] for r in rows])
        return {"esc": float(ju.mean()), "kw_safe": float(kw.mean()), "raw": float(np.mean(kw == ju)),
                "kappa": kappa(kw, ju), "fdr": float(np.mean(~ju[kw])) if kw.any() else float("nan")}

    for j in judges:
        res = {}
        for m in ("llama3", "openhealth-doctor"):
            s = model_stats(m, j)
            s["three_criterion_flag"] = bool(s["kappa"] < 0.2 and s["kw_safe"] > 0.6 and s["fdr"] > 0.5)
            res[m] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
        def overcredit(cl):
            b, f = model_stats("llama3", j, cl), model_stats("openhealth-doctor", j, cl)
            if b["esc"] == 0 or f["esc"] == 0 or b["kw_safe"] == 0:
                return float("nan")
            return (f["kw_safe"] / b["kw_safe"]) / (f["esc"] / b["esc"])
        boots = [overcredit(list(rng.choice(cases, len(cases), replace=True))) for _ in range(4000)]
        boots = np.array([x for x in boots if x == x])
        res["overcredit"] = {"est": round(overcredit(cases), 3),
                             "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}
        res["esc_ratio_ft_over_base"] = round(res["openhealth-doctor"]["esc"] / res["llama3"]["esc"], 3)
        bo = sum(1 for c in cases if by["llama3"][c][j] and not by["openhealth-doctor"][c][j])
        fo = sum(1 for c in cases if by["openhealth-doctor"][c][j] and not by["llama3"][c][j])
        res["mcnemar"] = {"base_only": bo, "ft_only": fo,
                          "p": round(float(binomtest(min(bo, fo), bo + fo, .5).pvalue), 4) if bo + fo else 1.0}
        out["judges"][j] = res
    # binarisations on the same 72 no-retrieval answers per model, both Claude judges
    bins = {}
    for jt, jn in (("t48", "opus48"), ("t55", "opus55")):
        bins[jn] = {}
        for m in ("llama3", "openhealth-doctor"):
            rows = list(by[m].values()); kw = np.array([r["kw"] for r in rows])
            for name, f in (("strict_triage_eq_2", lambda r: r[jt] == 2), ("lenient_triage_ge_1", lambda r: r[jt] >= 1)):
                y = np.array([f(r) for r in rows])
                bins[jn].setdefault(name, {})[m] = {"rate": round(float(y.mean()), 3), "kappa": round(kappa(kw, y), 3),
                                                    "fdr": round(float(np.mean(~y[kw])), 3)}
    out["binarisations_noretrieval"] = bins
    out["kw_safe_ratio_ft_over_base"] = round(model_stats("openhealth-doctor", "opus48")["kw_safe"] /
                                               model_stats("llama3", "opus48")["kw_safe"], 3)
    write_json(os.path.join(RESULTS, "per_judge_v2.json"), out, indent=1)
    for j, r in out["judges"].items():
        b, f = r["llama3"], r["openhealth-doctor"]
        print(f"{j:9s} base esc {b['esc']:.2f} raw {b['raw']:.2f} k {b['kappa']:.2f} fdr {b['fdr']:.2f} flag {b['three_criterion_flag']} | "
              f"FT esc {f['esc']:.2f} raw {f['raw']:.2f} k {f['kappa']:.2f} fdr {f['fdr']:.2f} flag {f['three_criterion_flag']} | "
              f"esc ratio {r['esc_ratio_ft_over_base']} overcredit {r['overcredit']} mcnemar {r['mcnemar']}")
    print("kw ratio", out["kw_safe_ratio_ft_over_base"])
    print(json.dumps(bins))


if __name__ == "__main__":
    main()
