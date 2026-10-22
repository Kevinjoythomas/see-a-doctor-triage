"""Merge the October judging runs (judged_main2a/b/c.json) into judged_main2.json (deduplicated by uid)."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, read_json, write_json
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(RESULTS, "judged_main2?.json"))):
    for r in read_json(f):
        if r["uid"] not in seen:
            seen.add(r["uid"]); rows.append(r)
    print(os.path.basename(f), "->", len(rows))
write_json(os.path.join(RESULTS, "judged_main2.json"), rows)
