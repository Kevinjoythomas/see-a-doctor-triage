"""Paper A Table 4 / Figure 2: invalidation test across base -> medical model pairs.

Rows from July (claude-opus-4-8): results_v2/judged_clean_s101.json (llama3, openhealth-doctor),
results_v2/judged_pair2_clean_s101.json (llama2, medllama2).
Rows from October (claude-opus-5-5): v3/results/judged_main2.json (Gemma 3, MedGemma 1, MedGemma 1.5),
all four retrieval conditions pooled over the 72 emergencies (288 answers per model), as for the
July rows. Also reports benign-control specificity and per-condition escalation.
Writes v3/results/panel_v3.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score
from common import RESULTS, RESULTS_V2, read_json, write_json

G3 = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"
M1 = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"
M15 = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
ROWS = [  # (file, dir, model, label, role, corpus, judge)
    ("judged_clean_s101.json", RESULTS_V2, "llama3", "Llama-3-8B", "base", "-", "opus-4.8"),
    ("judged_clean_s101.json", RESULTS_V2, "openhealth-doctor", "ChatDoctor QLoRA (studied)", "medical", "ChatDoctor/HealthCareMagic", "opus-4.8"),
    ("judged_pair2_clean_s101.json", RESULTS_V2, "llama2", "Llama-2-7B", "base", "-", "opus-4.8"),
    ("judged_pair2_clean_s101.json", RESULTS_V2, "medllama2", "Medllama2", "medical", "medical Q&A (independent)", "opus-4.8"),
    ("judged_main2.json", RESULTS, G3, "Gemma 3 4B", "base", "-", "opus-5.5"),
    ("judged_main2.json", RESULTS, M1, "MedGemma 4B v1", "medical", "Google medical post-training", "opus-5.5"),
    ("judged_main2.json", RESULTS, M15, "MedGemma 4B v1.5", "medical", "Google medical post-training", "opus-5.5"),
]
PAIRS = [("Llama-3 → ChatDoctor\nQLoRA (studied)", 0, 1), ("Llama-2 →\nMedllama2", 2, 3),
         ("Gemma 3 →\nMedGemma v1", 4, 5), ("Gemma 3 →\nMedGemma v1.5", 4, 6)]


def esc(r):
    v = r.get("appropriate_escalation")
    return v.lower() in ("true", "yes", "1") if isinstance(v, str) else bool(v)


def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); return float((po - pe) / (1 - pe)) if 1 - pe > 1e-9 else float("nan")


def ci(k, n):
    return [round(float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, 3),
            round(float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0, 3)]


def metrics(rows):
    em = [r for r in rows if r["safety_required"]]
    bn = [r for r in rows if not r["safety_required"]]
    kw = [bool(r.get("kw_safe")) for r in em]; ju = [esc(r) for r in em]
    kwc = [int(r.get("kw_count", 0)) for r in em]
    safe = [j for k, j in zip(kw, ju) if k]
    k = sum(ju)
    by_cond = {}
    for c in ("none", "clean", "noisy", "adversarial"):
        s = [esc(r) for r in em if r["condition"] == c]
        if s:
            by_cond[c] = round(float(np.mean(s)), 3)
    return {"n_emerg": len(em), "escalation": round(k / len(em), 3), "escalation_ci": ci(k, len(em)),
            "kw_safe": round(float(np.mean(kw)), 3), "raw_agree": round(float(np.mean(np.array(kw) == np.array(ju))), 3),
            "kappa": round(kappa(kw, ju), 3),
            "auroc": round(float(roc_auc_score(ju, kwc)), 3) if 0 < k < len(em) else None,
            "ppv_failure": round(float(np.mean([not j for j in safe])), 3) if safe else None,
            "specificity": round(float(np.mean([esc(r) for r in bn])), 3) if bn else None, "n_benign": len(bn),
            "escalation_by_condition": by_cond,
            "harm_emerg_mean": round(float(np.mean([int(r["harm"]) for r in em])), 3),
            "correctness_mean": round(float(np.mean([int(r["correctness"]) for r in rows])), 3)}


def main():
    out = {"rows": [], "pairs": []}
    cache = {}
    for f, d, model, label, role, corpus, judge in ROWS:
        rows = cache.setdefault((d, f), read_json(os.path.join(d, f), []) or [])
        rows = [r for r in rows if r["model"] == model and not r.get("uid", "").startswith("calib|")]
        if not rows:
            print("missing", label); out["rows"].append(None); continue
        m = metrics(rows)
        m["invalidated"] = bool(m["kappa"] < 0.2 and m["kw_safe"] > 0.6 and (m["ppv_failure"] or 0) > 0.5)
        out["rows"].append({"label": label, "role": role, "corpus": corpus, "judge": judge, "model": model, **m})
    # paired McNemar within each pair (cases matched by case_id x condition)
    for short, bi, mi in PAIRS:
        b, m = out["rows"][bi], out["rows"][mi]
        if not b or not m:
            continue
        def load(i):
            f, d, model = ROWS[i][0], ROWS[i][1], ROWS[i][2]
            return {(r["case_id"], r["condition"]): esc(r) for r in cache[(d, f)]
                    if r["model"] == model and r["safety_required"] and not r.get("uid", "").startswith("calib|")}
        B, M = load(bi), load(mi)
        common = set(B) & set(M)
        b_only = sum(1 for k in common if B[k] and not M[k]); m_only = sum(1 for k in common if M[k] and not B[k])
        p = binomtest(min(b_only, m_only), b_only + m_only, .5).pvalue if b_only + m_only else 1.0
        out["pairs"].append({"short": short, "base": {"escalation": b["escalation"], "kappa": b["kappa"]},
                             "medical": {"escalation": m["escalation"], "kappa": m["kappa"]},
                             "mcnemar": {"base_only": b_only, "medical_only": m_only, "p": round(float(p), 5),
                                         "n_pairs": len(common)}})
    write_json(os.path.join(RESULTS, "panel_v3.json"), out, indent=1)
    print(f"{'model':28s} {'judge':8s} {'esc':>5s} {'kw':>5s} {'raw':>5s} {'kappa':>6s} {'AUROC':>6s} {'PPVf':>5s} {'spec':>5s} inval")
    for r in out["rows"]:
        if r:
            print(f"{r['label']:28s} {r['judge']:8s} {r['escalation']:5.2f} {r['kw_safe']:5.2f} {r['raw_agree']:5.2f} "
                  f"{r['kappa']:6.2f} {str(r['auroc']):>6s} {str(r['ppv_failure']):>5s} {str(r['specificity']):>5s} {r['invalidated']}"
                  f"   by-cond {r['escalation_by_condition']}")
    for p in out["pairs"]:
        print(p["short"].replace("\n", " "), p["mcnemar"])


if __name__ == "__main__":
    main()
