"""Statistics for Paper B (reviewer requests). Writes v3/results/stats_B.json.
- generator escalation/specificity with Clopper-Pearson CIs (no retrieval / with retrieval)
- paired McNemar (escalation) and Wilcoxon (harm, correctness) for LoRA vs base and Gemma 3 vs MedGemma
- retrieval effect on the injected subset (McNemar per model) + the injected-chunk source mix
- test-retest: on cases where the cut-off blocked injection the none and clean prompts are identical;
  count differing answer texts and flipped escalation labels
- HealthBench emergency-referral composition (482 -> 453 -> 273; multi-turn; non-ASCII-heavy)
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta, binomtest, wilcoxon
from common import RESULTS, RESULTS_V2, read_json, write_json

G3 = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"; M1 = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"
M15 = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
E = lambda r: r["appropriate_escalation"] in (True, "true", 1)


def ci(k, n):
    return [round(float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, 3), round(float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0, 3)]


def mcn(a, b):
    c = set(a) & set(b); ao = sum(1 for x in c if a[x] and not b[x]); bo = sum(1 for x in c if b[x] and not a[x])
    return {"a_only": ao, "b_only": bo, "p": round(float(binomtest(min(ao, bo), ao + bo, .5).pvalue), 4) if ao + bo else 1.0}


def main():
    rows = read_json(os.path.join(RESULTS, "judged_calib_none.json")) + read_json(os.path.join(RESULTS, "judged_main2.json"))
    get = lambda m, c: {r["case_id"]: r for r in rows if r["model"] == m and r["condition"] == c}
    out = {"generators": {}}
    for m in ("llama3", "openhealth-doctor", G3, M1, M15):
        g = {}
        for c in ("none", "clean"):
            d = get(m, c)
            if not d:
                continue
            em = [E(r) for r in d.values() if r["safety_required"]]; bn = [E(r) for r in d.values() if not r["safety_required"]]
            g[c] = {"esc": round(np.mean(em), 3), "esc_ci": ci(sum(em), len(em)), "spec": round(np.mean(bn), 3),
                    "spec_ci": ci(sum(bn), len(bn)), "n": [len(em), len(bn)]}
        out["generators"][m] = g
    tests = {}
    for a, b in (("llama3", "openhealth-doctor"), (G3, M1), (G3, M15), ("openhealth-doctor", M15)):
        A, B = get(a, "none"), get(b, "none")
        ea = {k: E(v) for k, v in A.items() if v["safety_required"]}; eb = {k: E(v) for k, v in B.items() if v["safety_required"]}
        common = sorted(set(A) & set(B))
        em_c = [k for k in common if A[k]["safety_required"]]
        h = (lambda x, y: round(float(wilcoxon(x, y).pvalue), 4) if any(i != j for i, j in zip(x, y)) else 1.0)
        tests[f"{a} vs {b}"] = {"mcnemar_escalation": mcn(ea, eb),
                                "wilcoxon_harm_emerg": h([int(A[k]["harm"]) for k in em_c], [int(B[k]["harm"]) for k in em_c]),
                                "wilcoxon_correctness_all": h([int(A[k]["correctness"]) for k in common], [int(B[k]["correctness"]) for k in common])}
    out["paired_tests_noretrieval"] = tests
    ctx = read_json(os.path.join(RESULTS_V2, "contexts_cache_clean.json"))
    inj = {k for k, v in ctx.items() if v["clean"].get("text")}
    bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
    docs = read_json(os.path.join(RESULTS_V2, "corpus_docs.json"))
    src = {str(d["i"]): d["source"].split("\\")[-1] for d in docs}
    chunk_src = Counter(); chunk_ids = Counter()
    for k in inj:
        if not bench[k]["safety_required"]:
            continue
        ch = ctx[k]["clean"]["chunks"]
        if isinstance(ch, str):
            ch = eval(ch)
        top = ch[0]
        chunk_src[src.get(str(top.get("doc_index")), "?")] += 1
        chunk_ids[str(top.get("doc_index"))] += 1
    retr = {"n_injected_cases": len(inj), "n_injected_emergencies": sum(bench[k]["safety_required"] for k in inj),
            "top_chunk_source_counts": dict(chunk_src.most_common()), "most_common_top_chunk": chunk_ids.most_common(1)}
    for m in ("openhealth-doctor", G3, M1, M15):
        n0, c0 = get(m, "none"), get(m, "clean")
        a = {k: E(n0[k]) for k in inj if k in n0 and n0[k]["safety_required"]}
        b = {k: E(c0[k]) for k in inj if k in c0 and c0[k]["safety_required"]}
        retr[m] = {"none": sum(a.values()), "clean": sum(b.values()), "n": len(a), "mcnemar_none_vs_clean": mcn(a, b)}
    out["retrieval_injected_subset"] = retr
    # test-retest on identical prompts (blocked injection): compare raw texts from the grids + judged labels
    grids = {}
    for f in ("grid_gemmafam_clean_s101.json", "grid_medgemma15_clean_s101.json"):
        for m, cells in (read_json(os.path.join(RESULTS, f)) or {}).items():
            grids[m] = cells
    tr = {}
    for m in (G3, M1, M15):
        cells = grids.get(m, {})
        blocked = [k for k in bench if k not in inj]
        same_text = sum(1 for k in blocked if cells.get(f"{k}|none", {}).get("answer") == cells.get(f"{k}|clean", {}).get("answer"))
        n0, c0 = get(m, "none"), get(m, "clean")
        flips = sum(1 for k in blocked if k in n0 and k in c0 and E(n0[k]) != E(c0[k]))
        tr[m] = {"n_identical_prompts": len(blocked), "identical_answer_text": same_text, "escalation_label_flips": flips}
    out["test_retest_identical_prompts"] = tr
    # HealthBench composition
    hb = []
    with open(os.path.join(RESULTS_V2, "healthbench_raw.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "theme:emergency_referrals" in d.get("example_tags", []):
                hb.append(d)
    cat = Counter(next((t.split(":")[1] for t in d["example_tags"] if t.startswith("physician_agreed_category:")), "none") for d in hb)
    lab = [d for d in hb if any(t in d["example_tags"] for t in ("physician_agreed_category:emergent", "physician_agreed_category:non-emergent"))]
    def non_en(d):
        t = " ".join(x["content"] for x in d["prompt"] if x["role"] == "user")
        return sum(ord(ch) > 127 for ch in t) / max(1, len(t)) > 0.05
    out["healthbench"] = {"emergency_referrals_total": len(hb), "category_counts": dict(cat),
                          "labelled_emergent_or_nonemergent": len(lab),
                          "multi_turn_in_273": sum(1 for d in lab if len(d["prompt"]) > 1),
                          "mostly_non_ascii_in_273": sum(1 for d in lab if non_en(d))}
    write_json(os.path.join(RESULTS, "stats_B.json"), out, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "generators"}, indent=1))
    for m, g in out["generators"].items():
        print(m.split("/")[-1][:28], g)


if __name__ == "__main__":
    main()
