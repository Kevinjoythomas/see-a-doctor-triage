"""Judge drift + 4th-judge robustness on the 192 no-retrieval clean_s101 cells.
July judge = claude-opus-4-8 (results_v2/judged_clean_s101.json); new judge = the current
session judge (v3/results/judged_calib_none.json), same rubric, re-blinded, same answers.
Output: v3/results/calib_analysis.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import binomtest, beta
from sklearn.metrics import roc_auc_score
from common import RESULTS, RESULTS_V2, read_json, write_json

def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa*pb + (1-pa)*(1-pb); return float((po-pe)/(1-pe)) if 1-pe > 1e-9 else float("nan")
def ci(k, n): return [round(float(beta.ppf(.025,k,n-k+1)) if k else 0.0,3), round(float(beta.ppf(.975,k+1,n-k)) if k<n else 1.0,3)]
def wkappa(a, b, K):  # quadratic weighted kappa for ordinal scores
    a, b = np.array(a), np.array(b); O = np.zeros((K,K))
    for x,y in zip(a,b): O[x,y]+=1
    W = np.array([[(i-j)**2/(K-1)**2 for j in range(K)] for i in range(K)])
    E = np.outer(O.sum(1), O.sum(0))/O.sum()
    return float(1 - (W*O).sum()/(W*E).sum())

rows = read_json(os.path.join(RESULTS, "judged_calib_none.json"))
esc = lambda v: v in (True, "true", 1)
out = {"n": len(rows)}
new = [esc(r["appropriate_escalation"]) for r in rows]; old = [esc(r["july"]["appropriate_escalation"]) for r in rows]
out["drift_all"] = {"agree": round(float(np.mean(np.array(new)==np.array(old))),3), "kappa_escalation": round(kappa(new, old),3),
    "wkappa_triage": round(wkappa([r["triage"] for r in rows],[r["july"]["triage"] for r in rows],3),3),
    "wkappa_correctness": round(wkappa([r["correctness"] for r in rows],[r["july"]["correctness"] for r in rows],4),3),
    "wkappa_harm": round(wkappa([r["harm"] for r in rows],[r["july"]["harm"] for r in rows],3),3)}
per = {}
for m in ("llama3", "openhealth-doctor"):
    em = [r for r in rows if r["model"]==m and r["safety_required"]]
    bn = [r for r in rows if r["model"]==m and not r["safety_required"]]
    kn = sum(esc(r["appropriate_escalation"]) for r in em); ko = sum(esc(r["july"]["appropriate_escalation"]) for r in em)
    kw = [r["kw_safe"] for r in em]; ju = [esc(r["appropriate_escalation"]) for r in em]
    kws = [j for k,j in zip(kw,ju) if k]
    per[m] = {"n_emerg": len(em), "esc_new": round(kn/len(em),3), "esc_new_ci": ci(kn,len(em)), "esc_july": round(ko/len(em),3),
              "spec_new": round(np.mean([esc(r["appropriate_escalation"]) for r in bn]),3), "n_benign": len(bn),
              "kw_safe": round(float(np.mean(kw)),3), "kappa_kw_vs_new": round(kappa(kw, ju),3),
              "auroc_kwcount_new": round(float(roc_auc_score(ju,[r["kw_count"] for r in em])),3) if 0<sum(ju)<len(ju) else None,
              "ppv_failure_new": round(float(np.mean([not j for j in kws])),3) if kws else None,
              "drift_kappa_emerg": round(kappa([esc(r["appropriate_escalation"]) for r in em],[esc(r["july"]["appropriate_escalation"]) for r in em]),3)}
out["per_model"] = per
bc = {r["case_id"]: esc(r["appropriate_escalation"]) for r in rows if r["model"]=="llama3" and r["safety_required"]}
fc = {r["case_id"]: esc(r["appropriate_escalation"]) for r in rows if r["model"]=="openhealth-doctor" and r["safety_required"]}
b_only = sum(1 for c in bc if bc[c] and not fc.get(c)); f_only = sum(1 for c in fc if fc[c] and not bc.get(c))
out["mcnemar_new"] = {"base_only": b_only, "ft_only": f_only, "p": round(binomtest(min(b_only,f_only), b_only+f_only, .5).pvalue,4)}
# 4-judge Fleiss kappa on the 144 emergency cells (claude-4.8, qwen, mistral, new judge)
maj = {r["uid"].replace("clean_s101|","calib|"): r for r in read_json(os.path.join(RESULTS_V2, "majority_clean_s101.json"))}
mat = []
for r in rows:
    m = maj.get(r["uid"])
    if m: mat.append([bool(m["claude"]), bool(m["qwen"]), bool(m["mistral"]), esc(r["appropriate_escalation"])])
mat = np.array(mat, float)
def fleiss(M):
    n = M.shape[1]; p1 = M.mean(); P = ((M.sum(1)**2 + (n-M.sum(1))**2) - n)/(n*(n-1))
    Pe = p1**2 + (1-p1)**2; return float((P.mean()-Pe)/(1-Pe))
out["fleiss_3judges_july"] = round(fleiss(mat[:, :3]), 3); out["fleiss_4judges"] = round(fleiss(mat), 3)
out["fleiss_n_cells"] = len(mat)
names = ["claude48","qwen","mistral","new"]
out["pairwise_kappa"] = {f"{names[i]}~{names[j]}": round(kappa(mat[:,i],mat[:,j]),3) for i in range(4) for j in range(i+1,4)}
write_json(os.path.join(RESULTS, "calib_analysis.json"), out, indent=1)
print(json.dumps(out, indent=1))
