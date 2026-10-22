"""Measure the 8B fine-tune's serving speed on this laptop (4 GB GPU, partial CPU offload),
under the same decoding settings as the grids, on the first 20 clean-benchmark questions
(no retrieval). Records tokens/s, seconds/answer, and the share of the model held in VRAM.
Writes v3/results/latency_ft.json.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import RESULTS, RESULTS_V2, SYSTEM_PROMPT, _post, read_json, write_json, unload

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openhealth-doctor"
cases = read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))[:20]
runs = []
_post("/api/generate", {"model": MODEL, "prompt": "Hello", "stream": False, "options": {"num_predict": 1}})
ps = json.loads(urllib.request.urlopen("http://localhost:11434/api/ps").read())
share = next(((m["size_vram"] / m["size"]) for m in ps["models"] if m["name"].startswith(MODEL)), None)
for c in cases:
    t = time.time()
    r = _post("/api/generate", {"model": MODEL, "prompt": f"{SYSTEM_PROMPT}\n\nPatient: {c['question']}\nDoctor:",
                                "stream": False, "options": {"temperature": 0.3, "num_predict": 300,
                                                             "num_ctx": 2048, "seed": 101}})
    runs.append({"s": time.time() - t, "tok": r.get("eval_count", 0), "dur": (r.get("eval_duration") or 1) / 1e9})
out = {"model": MODEL, "n": len(runs), "vram_share": round(share, 3) if share else None,
       "tok_per_s_median": round(float(np.median([x["tok"] / x["dur"] for x in runs])), 1),
       "s_per_answer_median": round(float(np.median([x["s"] for x in runs])), 2),
       "tokens_mean": round(float(np.mean([x["tok"] for x in runs])), 1)}
unload(MODEL)
prev = read_json(os.path.join(RESULTS, "latency_ft.json"), {}) or {}
prev[MODEL] = out
write_json(os.path.join(RESULTS, "latency_ft.json"), prev, indent=1)
print(out)
