"""Blind one or more grids (plus an optional July calibration sample) into judging batches.

Usage:
  python judge_prep.py <tag> <grid1.json>[:model1,model2] [<grid2.json>...] [--calib N]

- Grid paths are relative to v3/results/ unless absolute.
- --calib N adds N cells re-sampled from the July-judged results_v2/judged_clean_s101.json
  (stratified by model x condition x safety_required, deterministic) so the current judge
  can be compared with the July judge (claude-opus-4-8) on identical answers. They are
  blinded and shuffled together with the new cells; the judge cannot tell them apart.
Writes v3/results/judge_batches_<tag>/batch_XX.json and v3/results/blindmap_<tag>.json.
The judge sees exactly the v2 fields: blind_id, safety_required, category, question, answer.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, RESULTS_V2, read_json

B = 14


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def cells_from_grid(path, models=None, tag=""):
    grid = read_json(path)
    out = []
    for model, entries in grid.items():
        if not isinstance(entries, dict) or (models and model not in models):
            continue
        for c in entries.values():
            if not isinstance(c, dict) or not c.get("answer"):
                continue
            out.append({"uid": f"{tag}|{model}|{c['case_id']}|{c['condition']}",
                        "source": os.path.basename(path), "model": model,
                        "case_id": c["case_id"], "condition": c["condition"],
                        "safety_required": c.get("safety_required"), "category": c.get("category"),
                        "question": c.get("question", ""), "answer": c["answer"]})
    return out


def calib_cells(n):
    rows = read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))
    if n == -1:  # every no-retrieval cell (the 3-judge IRR subset + benign controls)
        pick = [r for r in rows if r["condition"] == "none"]
        return [{"uid": f"calib|{r['model']}|{r['case_id']}|{r['condition']}",
                 "source": "judged_clean_s101.json", "model": r["model"],
                 "case_id": r["case_id"], "condition": r["condition"],
                 "safety_required": r["safety_required"], "category": r.get("category"),
                 "question": r.get("question", ""), "answer": r["answer"],
                 "july": {k: r.get(k) for k in ("correctness", "triage", "harm", "calibration",
                                                 "holistic", "appropriate_escalation")}}
                for r in pick]
    strata = {}
    for r in rows:
        k = (r["model"], r["condition"], bool(r["safety_required"]))
        strata.setdefault(k, []).append(r)
    # n split over 2 models x 4 conds; within each, 3:1 safety:benign (mirrors 72:24)
    per = max(1, n // 8)
    n_safe = round(per * 0.75)
    pick = []
    for (model, cond, safe), rs in sorted(strata.items()):
        rs = sorted(rs, key=lambda r: h("calib|" + r["uid"]))
        pick += rs[: (n_safe if safe else per - n_safe)]
    out = []
    for r in pick:
        out.append({"uid": f"calib|{r['model']}|{r['case_id']}|{r['condition']}",
                    "source": "judged_clean_s101.json", "model": r["model"],
                    "case_id": r["case_id"], "condition": r["condition"],
                    "safety_required": r["safety_required"], "category": r.get("category"),
                    "question": r.get("question", ""), "answer": r["answer"],
                    "july": {k: r.get(k) for k in ("correctness", "triage", "harm", "calibration",
                                                    "holistic", "appropriate_escalation")}})
    return out


def main():
    tag = sys.argv[1]
    args = sys.argv[2:]
    calib = 0
    if "--calib" in args:
        i = args.index("--calib")
        calib = -1 if args[i + 1] == "all_none" else int(args[i + 1])
        args = args[:i] + args[i + 2:]
    cells = []
    if args == ["-"]:
        args = []
    for spec in args:
        path, _, models = spec.partition(":")
        path = path if os.path.isabs(path) else os.path.join(RESULTS, path)
        cells += cells_from_grid(path, set(models.split(",")) if models else None,
                                 tag=os.path.splitext(os.path.basename(path))[0])
    if calib:
        cells += calib_cells(calib)
    if not cells:
        raise SystemExit("no cells")
    # question text is needed for calib rows: July judged rows may lack it -> fill from benchmark
    bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
    for c in cells:
        if not c["question"] and c["case_id"] in bench:
            c["question"] = bench[c["case_id"]]["question"]
            c["category"] = c["category"] or bench[c["case_id"]]["category"]
    cells.sort(key=lambda r: h(tag + "|" + r["uid"]))
    bdir = os.path.join(RESULTS, f"judge_batches_{tag}")
    os.makedirs(bdir, exist_ok=True)
    for f in os.listdir(bdir):
        os.remove(os.path.join(bdir, f))
    blindmap, nb = {}, 0
    for bi in range(0, len(cells), B):
        batch = []
        for j, r in enumerate(cells[bi:bi + B]):
            bid = f"V{bi // B:03d}{j:02d}"
            blindmap[bid] = {k: r[k] for k in r if k not in ("question",)}
            batch.append({"blind_id": bid, "safety_required": r["safety_required"],
                          "category": r["category"], "question": r["question"],
                          "answer": r["answer"]})
        json.dump(batch, open(os.path.join(bdir, f"batch_{bi // B:02d}.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)
        nb += 1
    json.dump(blindmap, open(os.path.join(RESULTS, f"blindmap_{tag}.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=0)
    n_cal = sum(1 for c in cells if c["uid"].startswith("calib|"))
    print(f"tag={tag}: {len(cells)} cells ({n_cal} calibration) -> {nb} batches in {bdir}")
    print(f"N_BATCHES={nb}")


if __name__ == "__main__":
    main()
