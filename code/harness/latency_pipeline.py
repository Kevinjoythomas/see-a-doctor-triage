"""End-to-end latency and memory of the deployed triage path on the 4 GB laptop (Paper B, second review round).

For 16 benchmark messages (8 emergencies, 8 controls) each configuration runs the real sequence
  red-flag rules -> triage call (Gemma 3 4B) -> generation (300 tokens max)
and records per-message wall time of the triage call, the generation and the total, the model-load time Ollama
reports, and the lowest MemAvailable seen on the host (sampled every 0.25 s). Retrieval is not included.

Configurations
  split        Gemma 3 gate with log-probabilities (GPU) + MedGemma 1.5 generator (GPU): models swap per message
  split_label  Gemma 3 gate, label only (no log-probabilities) + MedGemma 1.5: models swap per message
  single       Gemma 3 as gate (log-probabilities) and generator: no swap
  cpu_label    Gemma 3 label-only gate on the CPU (num_gpu 0) + MedGemma 1.5 on the GPU: no GPU swap

Usage: python latency_pipeline.py [config ...]   -> v3/results/latency_pipeline.json
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import RESULTS, RESULTS_V2, _post, log, read_json, unload, write_json
from gate import GATE_OPTS, TRIAGE_PROMPT, _p_emergency, _parse
from gen_grid import build_prompt
import redflags

G3 = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"
M15 = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
GEN_OPTS = {"temperature": 0.3, "num_predict": 300, "num_ctx": 2048, "seed": 101, "top_k": 64, "top_p": 0.95, "num_gpu": 99}
CONFIGS = {"split": (G3, True, 99, M15), "split_label": (G3, False, 99, M15),
           "single": (G3, True, 99, G3), "cpu_label": (G3, False, 0, M15)}
THR = 2.4810910014553345e-05


class MemWatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.min_avail = None; self.stop = False

    def run(self):
        while not self.stop:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        v = int(line.split()[1]) // 1024
                        self.min_avail = v if self.min_avail is None else min(self.min_avail, v)
            time.sleep(0.25)


def gate_call(model, text, use_lp, num_gpu):
    body = {"model": model, "prompt": TRIAGE_PROMPT.format(text=text), "stream": False, "keep_alive": "30m",
            "options": {**GATE_OPTS, "num_gpu": num_gpu}}
    if use_lp:
        body.update({"logprobs": True, "top_logprobs": 10})
    t = time.time(); r = _post("/api/generate", body, timeout=600); dt = time.time() - t
    lv = _parse(r.get("response", ""))
    fired = (_p_emergency(r) or 0) >= THR if use_lp else lv == "EMERGENCY"
    return fired, dt, (r.get("load_duration") or 0) / 1e9


def gen_call(model, text):
    body = {"model": model, "prompt": build_prompt(text), "stream": False, "keep_alive": "30m", "options": GEN_OPTS}
    t = time.time(); r = _post("/api/generate", body, timeout=600); dt = time.time() - t
    return dt, (r.get("load_duration") or 0) / 1e9, r.get("eval_count") or 0


def main():
    which = sys.argv[1:] or list(CONFIGS)
    bench = read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))
    em = [c for c in bench if c["safety_required"]][::9]; bn = [c for c in bench if not c["safety_required"]][::3]
    msgs = em + bn
    path = os.path.join(RESULTS, "latency_pipeline.json")
    out = read_json(path, {}) or {}
    with open("/proc/meminfo") as f:
        total = int(next(l for l in f if l.startswith("MemTotal")).split()[1]) // 1024
    for name in which:
        gm, use_lp, ngpu, genm = CONFIGS[name]
        for m in {gm, genm}:
            unload(m)
        time.sleep(3)
        w = MemWatch(); w.start()
        rows = []
        for i, c in enumerate(msgs):
            t0 = time.time()
            rule = redflags.detect(c["question"])
            rule_fired = bool(rule[0]) if isinstance(rule, tuple) else bool(rule)
            fired, tg, lg = gate_call(gm, c["question"], use_lp, ngpu)
            tgen, lgen, ntok = gen_call(genm, c["question"])
            rows.append({"case_id": c["id"], "emergency": c["safety_required"], "fired": bool(fired or rule_fired),
                         "gate_s": round(tg, 2), "gate_load_s": round(lg, 2), "gen_s": round(tgen, 2),
                         "gen_load_s": round(lgen, 2), "gen_tokens": ntok, "total_s": round(time.time() - t0, 2)})
            log(f"latency {name} {i+1}/{len(msgs)} total {rows[-1]['total_s']}s")
        w.stop = True; time.sleep(0.5)
        tot = np.array([r["total_s"] for r in rows]); gs = np.array([r["gate_s"] for r in rows])
        out[name] = {"gate_model": gm, "logprobs": use_lp, "gate_num_gpu": ngpu, "generator": genm, "n": len(rows),
                     "total_median_s": round(float(np.median(tot)), 1), "total_p95_s": round(float(np.percentile(tot, 95)), 1),
                     "gate_median_s": round(float(np.median(gs)), 2), "gate_p95_s": round(float(np.percentile(gs, 95)), 2),
                     "gen_median_s": round(float(np.median([r["gen_s"] for r in rows])), 1),
                     "median_load_s_per_message": round(float(np.median([r["gate_load_s"] + r["gen_load_s"] for r in rows])), 1),
                     "host_mem_total_mb": total, "host_min_available_mb": w.min_avail,
                     "gate_fired_emerg": sum(r["fired"] for r in rows if r["emergency"]),
                     "gate_fired_controls": sum(r["fired"] for r in rows if not r["emergency"]), "rows": rows}
        write_json(path, out, indent=1)
        log(f"latency {name}: median {out[name]['total_median_s']} s, p95 {out[name]['total_p95_s']} s, "
            f"min avail {w.min_avail} MB")
        for m in {gm, genm}:
            unload(m)


if __name__ == "__main__":
    main()
