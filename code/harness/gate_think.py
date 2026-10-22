"""Repair the MedGemma 1.5 gate run. MedGemma 1.5 sometimes opens with a visible reasoning trace
("thought\\n..."), so the 6-token budget of gate.py returned no label (UNPARSED) for 312 of 649 items.
Here every UNPARSED item is re-asked with the same prompt and room to finish (num_predict 600), and the
model's final label is taken as the LAST occurrence of EMERGENCY / URGENT / ROUTINE in its reply.
Because those items have no first-token probability, p_emergency is cleared for ALL items of this model,
so its candidates are evaluated on the discrete label only (consistent across items).
Options match the concurrent HealthBench generation (num_ctx 4096, num_gpu 99) so the runner is shared.
Usage: python gate_think.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, _post, log, read_json, write_json
from gate import TRIAGE_PROMPT, load_sets

M = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
PATH = os.path.join(RESULTS, "gate_scores_llm_hf.co_unsloth_medgemma-1.5-4b-it-GGUF_Q4_K_M.json")
LV = re.compile(r"\b(EMERGENCY|URGENT|ROUTINE)\b", re.I)


def main():
    d = read_json(PATH)
    sets = load_sets()
    items = {s: {it["id"]: it for it in v} for s, v in sets.items()}
    todo = [(s, i) for s in d for i, v in d[s].items() if v.get("level") in ("UNPARSED", "ERROR")]
    log(f"gate_think: {len(todo)} items to re-ask")
    for n, (s, i) in enumerate(todo):
        text = items[s][i]["convo_text"]
        if len(text) > 5000:
            text = "[...earlier conversation omitted...]\n" + text[-5000:]
        body = {"model": M, "prompt": TRIAGE_PROMPT.format(text=text), "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 600, "num_ctx": 4096, "seed": 0, "num_gpu": 99,
                            "top_k": 64, "top_p": 0.95, "min_p": 0.0}}
        for attempt in range(4):
            try:
                r = _post("/api/generate", body, timeout=300); break
            except Exception as e:
                if attempt == 3:
                    r = {"response": f"ERROR {e}"}
                import time; time.sleep(10 * (attempt + 1))
        txt = r.get("response", "")
        found = LV.findall(txt)
        d[s][i].update({"level": found[-1].upper() if found else "UNPARSED", "raw": txt[-120:],
                        "reasked_with_thinking": True})
        if n % 20 == 0:
            write_json(PATH, d); log(f"gate_think {n}/{len(todo)}")
    for s in d:
        for v in d[s].values():
            v["p_emergency"] = None
    write_json(PATH, d)
    from collections import Counter
    log(f"gate_think done: " + str({s: dict(Counter(v['level'] for v in d[s].values())) for s in d}))


if __name__ == "__main__":
    main()
