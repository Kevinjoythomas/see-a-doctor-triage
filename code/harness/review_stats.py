"""Statistics requested by the v3 hostile review of Paper A. Writes v3/results/review_stats.json.
(a) case-clustered bootstrap 95% CIs (cluster = case, 5,000 draws) for keyword-vs-judge raw agreement,
    kappa, AUROC, P(under | keyword-safe) per model and for base-minus-FT differences (July judge, 288/model)
(b) no-retrieval escalation per seed (101/102/103) and per judge version
(c) judge drift restricted to the 144 emergency answers
(d) share of under-triaging answers containing >= 1 rubric keyword
(e) answer length vs escalation: medians and Mann-Whitney per model (no-retrieval, July judge)
(f) MCQ: paired McNemar where per-item results exist, else two-proportion test
(g) word-boundary variant of the keyword rubric (the harness uses case-insensitive substring matching)
(h) best-threshold kappa on keyword COUNT (does re-thresholding recover validity?)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.metrics import roc_auc_score
from common import RESULTS, RESULTS_V2, SAFETY_KW, read_json, write_json

rng = np.random.default_rng(20261022)
esc = lambda r: r["appropriate_escalation"] in (True, "true", 1)


def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); return float((po - pe) / (1 - pe)) if 1 - pe > 1e-9 else float("nan")


def stats(rows):
    kw = np.array([bool(r["kw_safe"]) for r in rows]); ju = np.array([esc(r) for r in rows])
    kc = np.array([int(r["kw_count"]) for r in rows])
    out = {"raw": float(np.mean(kw == ju)), "kappa": kappa(kw, ju),
           "p_under_given_safe": float(np.mean(~ju[kw])) if kw.any() else float("nan"),
           "auroc": float(roc_auc_score(ju, kc)) if 0 < ju.sum() < len(ju) else float("nan")}
    return out


def boot(rows_by_model, B=5000):
    cases = sorted({r["case_id"] for rs in rows_by_model.values() for r in rs})
    idx = {m: {} for m in rows_by_model}
    for m, rs in rows_by_model.items():
        for r in rs:
            idx[m].setdefault(r["case_id"], []).append(r)
    samples = {m: [] for m in rows_by_model}; diffs = []
    for _ in range(B):
        pick = rng.choice(cases, len(cases), replace=True)
        cur = {}
        for m in rows_by_model:
            rs = [r for c in pick for r in idx[m].get(c, [])]
            cur[m] = stats(rs); samples[m].append(cur[m])
        diffs.append({k: cur["llama3"][k] - cur["openhealth-doctor"][k] for k in cur["llama3"]})
    def ci(vals):
        v = np.array([x for x in vals if x == x])
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    res = {}
    for m in rows_by_model:
        pt = stats(rows_by_model[m])
        res[m] = {k: {"est": round(pt[k], 3), "ci95": ci([s[k] for s in samples[m]])} for k in pt}
    res["base_minus_ft"] = {k: ci([d[k] for d in diffs]) for k in diffs[0]}
    return res


def main():
    out = {}
    july = read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))
    em = {m: [r for r in july if r["model"] == m and r["safety_required"]] for m in ("llama3", "openhealth-doctor")}
    out["bootstrap_pooled_288"] = boot(em)
    em0 = {m: [r for r in rs if r["condition"] == "none"] for m, rs in em.items()}
    out["bootstrap_noretrieval_72"] = boot(em0)
    # (b) per seed, no retrieval
    seeds = {}
    for s in ("s101", "s102", "s103"):
        rows = read_json(os.path.join(RESULTS_V2, f"judged_clean_{s}.json"), []) or []
        d = {}
        for m in ("llama3", "openhealth-doctor"):
            x = [esc(r) for r in rows if r["model"] == m and r["safety_required"] and r["condition"] == "none"]
            allc = [esc(r) for r in rows if r["model"] == m and r["safety_required"]]
            d[m] = {"none": round(float(np.mean(x)), 3) if x else None, "n_none": len(x),
                    "pooled": round(float(np.mean(allc)), 3) if allc else None, "n_pooled": len(allc)}
        d["n_graded_total"] = len(rows)
        seeds[s] = d
    out["per_seed"] = seeds
    # (c) drift on emergencies only
    cal = read_json(os.path.join(RESULTS, "judged_calib_none.json"))
    e = [r for r in cal if r["safety_required"]]
    a = [esc(r) for r in e]; b = [r["july"]["appropriate_escalation"] in (True, "true", 1) for r in e]
    flips = {m: {"july_yes_new_no": sum(1 for r in e if r["model"] == m and r["july"]["appropriate_escalation"] and not esc(r)),
                 "july_no_new_yes": sum(1 for r in e if r["model"] == m and not r["july"]["appropriate_escalation"] and esc(r))}
             for m in ("llama3", "openhealth-doctor")}
    out["drift_emergencies_only"] = {"n": len(e), "agree": round(float(np.mean(np.array(a) == np.array(b))), 3),
                                     "kappa": round(kappa(a, b), 3), "flips": flips}
    # (d) keyword share among under-triaging answers (pooled 288)
    out["kw_share_in_undertriage"] = {m: round(float(np.mean([bool(r["kw_safe"]) for r in rs if not esc(r)])), 3)
                                      for m, rs in em.items()}
    # (e) length
    L = {}
    for m, rs in em0.items():
        wl = lambda r: len(r["answer"].split())
        ya = [wl(r) for r in rs if esc(r)]; na = [wl(r) for r in rs if not esc(r)]
        p = mannwhitneyu(ya, na).pvalue if ya and na else None
        L[m] = {"median_words_escalating": float(np.median(ya)) if ya else None,
                "median_words_undertriage": float(np.median(na)) if na else None,
                "n_esc": len(ya), "n_under": len(na), "mannwhitney_p": round(float(p), 3) if p else None}
    out["length_noretrieval"] = L
    # (f) MCQ
    mcq = read_json(os.path.join(RESULTS_V2, "mcq_results.json"))
    mq = {}
    for bench in ("medqa", "pubmedqa"):
        b0, f0 = mcq[f"{bench}|base"], mcq[f"{bench}|finetuned"]
        kb, kf, n = round(b0["accuracy"] * b0["n"]), round(f0["accuracy"] * f0["n"]), b0["n"]
        chi = chi2_contingency([[kb, n - kb], [kf, n - kf]])
        mq[bench] = {"base": b0["accuracy"], "ft": f0["accuracy"], "n": n, "p_two_proportion": round(float(chi[1]), 3),
                     "note": "per-item answers not stored; unpaired test"}
    out["mcq"] = mq
    # (g) word-boundary keyword variant (pooled 288)
    wb = re.compile(r"\b(" + "|".join(re.escape(k) for k in SAFETY_KW) + r")\b", re.I)
    g = {}
    for m, rs in em.items():
        kw = [bool(wb.search(r["answer"])) for r in rs]; ju = [esc(r) for r in rs]
        kws = [j for k, j in zip(kw, ju) if k]
        g[m] = {"kw_safe": round(float(np.mean(kw)), 3), "kappa": round(kappa(kw, ju), 3),
                "p_under_given_safe": round(float(np.mean([not x for x in kws])), 3)}
    out["word_boundary_variant"] = g
    # (h) best-threshold kappa on keyword count
    h = {}
    for m, rs in em.items():
        kc = np.array([int(r["kw_count"]) for r in rs]); ju = np.array([esc(r) for r in rs])
        best = max(((t, kappa(kc >= t, ju)) for t in range(1, 6)), key=lambda x: x[1])
        h[m] = {"best_threshold": int(best[0]), "best_kappa": round(best[1], 3)}
    out["best_threshold_kappa"] = h
    write_json(os.path.join(RESULTS, "review_stats.json"), out, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
