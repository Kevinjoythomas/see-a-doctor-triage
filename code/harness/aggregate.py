"""De-blind a v3 judging run: read v3/results/judge_out_<tag>/batch_*.json + blindmap_<tag>.json,
attach the keyword rubric computed from each answer, write v3/results/judged_<tag>.json.

Usage: python aggregate.py <tag>
Rows keep every blindmap field (uid, source, model, case_id, condition, safety_required,
category, answer, and 'july' labels for calibration cells) plus the new judge scores.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, SAFETY_KW, read_json, write_json


def main(tag):
    blind = read_json(os.path.join(RESULTS, f"blindmap_{tag}.json"))
    scores = {}
    files = sorted(glob.glob(os.path.join(RESULTS, f"judge_out_{tag}", "batch_*.json")))
    for f in files:
        d = read_json(f)
        for s in d.get("scores", []):
            scores[s["blind_id"]] = s
    rows, missing = [], []
    for bid, meta in blind.items():
        s = scores.get(bid)
        if s is None:
            missing.append(bid)
            continue
        ans = meta.get("answer", "")
        rows.append({**meta, **{k: v for k, v in s.items() if k != "blind_id"}, "blind_id": bid,
                     "kw_count": sum(k in ans.lower() for k in SAFETY_KW),
                     "kw_safe": any(k in ans.lower() for k in SAFETY_KW)})
    write_json(os.path.join(RESULTS, f"judged_{tag}.json"), rows)
    print(f"{tag}: {len(files)} batch files, {len(rows)} judged rows, {len(missing)} missing")
    if missing:
        print("missing blind ids:", missing[:20])


if __name__ == "__main__":
    main(sys.argv[1])
