"""Triage safety gate for OpenHealth (Paper B): detectors + offline evaluation.

Detectors (all score the PATIENT side of the conversation before any answer exists):
  rule   - regex red-flag lexicon (redflags.py), no training
  emb    - nomic-embed-text similarity to authored emergency vs non-urgent prototypes
  llm:M  - zero-shot triage call to local model M (EMERGENCY / URGENT / ROUTINE),
           with P(EMERGENCY) from token logprobs when Ollama returns them
Ensembles are formed in analysis (gate_analysis.py). Thresholds are chosen on the
DEV set only (corpus-adjacent 100 cases); test sets are scored once.

Usage:
  python gate.py rule
  python gate.py emb
  python gate.py llm <ollama_model> [--sets dev,clean,hb]
Outputs: v3/results/gate_scores_<detector>.json  {set: {item_id: {...}}}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import RESULTS, RESULTS_V2, _post, batch_embed, log, read_json, write_json, unload
import redflags

HB_RAW = os.path.join(RESULTS_V2, "healthbench_raw.jsonl")
HB_LABEL = {"physician_agreed_category:emergent": 1,
            "physician_agreed_category:non-emergent": 0,
            "physician_agreed_category:conditionally-emergent": 2}


def load_sets():
    """Return {set_name: [ {id, user_text, convo_text, label} ]}. label: 1 emergency,
    0 not, 2 conditionally-emergent (HealthBench only; reported separately)."""
    sets = {}
    orig = read_json(os.path.join(RESULTS_V2, "orig_cases.json"))
    sets["dev"] = [{"id": c["id"], "user_text": c["question"],
                    "convo_text": f"Patient: {c['question']}",
                    "label": int(bool(c["safety_required"])), "category": c["category"]}
                   for c in orig]
    clean = read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))
    sets["clean"] = [{"id": c["id"], "user_text": c["question"],
                      "convo_text": f"Patient: {c['question']}",
                      "label": int(bool(c["safety_required"])), "category": c["category"]}
                     for c in clean]
    hb = []
    with open(HB_RAW, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tags = d.get("example_tags", [])
            if "theme:emergency_referrals" not in tags:
                continue
            lab = next((HB_LABEL[t] for t in tags if t in HB_LABEL), None)
            if lab is None:
                continue
            turns = d["prompt"]
            user_text = "\n".join(t["content"] for t in turns if t["role"] == "user")
            convo = "\n".join(("Patient" if t["role"] == "user" else "Assistant") + ": " + t["content"]
                              for t in turns if t["role"] in ("user", "assistant"))
            hb.append({"id": d["prompt_id"], "user_text": user_text, "convo_text": convo,
                       "label": lab, "category": "hb_er", "n_turns": len(turns)})
    sets["hb"] = hb
    return sets


# ---------------------------------------------------------------- rule
def run_rule(sets):
    out = {}
    for name, items in sets.items():
        out[name] = {}
        for it in items:
            fired, hits = redflags.detect(it["user_text"])
            out[name][it["id"]] = {"label": it["label"], "fired": fired,
                                   "score": redflags.score(it["user_text"]), "hits": hits}
    write_json(os.path.join(RESULTS, "gate_scores_rule.json"), out)
    log("rule detector scored")


# ---------------------------------------------------------------- emb
EMERGENCY_PROTOTYPES = [
    "Crushing chest pain spreading to my left arm and jaw, I am sweating and short of breath.",
    "My father's face is drooping on one side, his arm is weak and his speech is slurred.",
    "I suddenly have the worst headache of my life, it came on like a thunderclap.",
    "My lips and tongue are swelling after eating peanuts and my throat feels like it is closing.",
    "I can't breathe properly, I'm gasping even when sitting still and my lips look blue.",
    "My baby is three weeks old and has a fever of 38.5 and is not feeding.",
    "My child is floppy, very hard to wake up and has a rash that doesn't fade when pressed.",
    "I took a whole bottle of paracetamol tablets an hour ago.",
    "I want to kill myself tonight and I have a plan.",
    "I am vomiting blood and feel faint.",
    "I'm pregnant, have severe one-sided pain in my lower abdomen and I'm bleeding.",
    "I am 34 weeks pregnant with a severe headache, blurred vision and swollen face.",
    "My blood sugar has been over 450, I keep vomiting and my breath smells fruity.",
    "He collapsed and is not responding, he is breathing strangely.",
    "My husband had a seizure that has lasted more than five minutes.",
    "Sudden severe pain in my testicle with swelling and vomiting.",
    "I have numbness between my legs and can't control my bladder after back pain.",
    "Fever, stiff neck, I can't stand the light and I'm getting confused.",
    "Sudden tearing pain in my chest going through to my back.",
    "My calf is swollen and painful and now I'm suddenly short of breath with chest pain.",
    "I fell and hit my head, I'm on warfarin and now I keep vomiting and feel drowsy.",
    "My elderly mother has a fever, is very confused, breathing fast and has mottled skin.",
    "My toddler swallowed a button battery.",
    "There is a deep cut on my leg that is spurting blood and won't stop bleeding.",
    "My blood pressure is 210/125 and I have chest pain and a pounding headache.",
    "The whole house smells of gas, we all have headaches and feel sick, the CO alarm went off.",
    "Black tarry stools and I feel dizzy and weak when I stand up.",
    "My heart is racing at 190 and I feel like I'm going to pass out.",
    "Sudden loss of vision in one eye that started an hour ago.",
    "Severe abdominal pain, my belly is rigid and I have a high fever.",
]
ROUTINE_PROTOTYPES = [
    "I've had a runny nose and mild sore throat for two days, no fever.",
    "What is a normal resting heart rate for someone my age?",
    "I have mild acne on my chin, what cream can I use?",
    "My knee aches a bit after running, it gets better with rest.",
    "How much vitamin D should I take in winter?",
    "I have occasional heartburn after spicy food.",
    "Is it safe to take ibuprofen with my multivitamin?",
    "I get a tension headache at the end of long work days that goes away with paracetamol.",
    "My toddler has a mild cough but is eating, playing and sleeping normally.",
    "I have dandruff and an itchy scalp.",
    "How can I improve my sleep? I wake up tired sometimes.",
    "I have a small itchy mosquito bite that is slightly red.",
    "What foods help lower cholesterol?",
    "I've had mild constipation for a few days.",
    "My period is three days late and I've been stressed, the pregnancy test is negative.",
    "I sprained my ankle lightly yesterday, I can walk on it with some pain.",
    "Can you explain what my HbA1c result of 5.9 means?",
    "I have a dry cough that has lingered for a week after a cold, otherwise I feel fine.",
    "Is it normal to have some muscle soreness two days after starting the gym?",
    "My eyes feel dry when I use my laptop for long hours.",
    "I have a cold sore on my lip.",
    "What is the difference between a cold and the flu?",
    "Mild back stiffness in the morning that loosens up after I move around.",
    "My child has a low fever of 37.8 after vaccines but is playful.",
    "I keep getting mouth ulcers when I'm stressed.",
    "I have a stuffy nose from hay fever every spring.",
    "How often should I get my blood pressure checked if it's normal?",
    "I have a rash from a new laundry detergent, it's itchy but not spreading.",
    "I feel a bit bloated after meals.",
    "Can stress cause occasional eye twitching?",
]


def run_emb(sets):
    pe = np.array(batch_embed(["search_query: " + t for t in EMERGENCY_PROTOTYPES]))
    pr = np.array(batch_embed(["search_query: " + t for t in ROUTINE_PROTOTYPES]))
    pe /= np.linalg.norm(pe, axis=1, keepdims=True)
    pr /= np.linalg.norm(pr, axis=1, keepdims=True)
    out = {}
    for name, items in sets.items():
        out[name] = {}
        texts = [it["user_text"][:2000] for it in items]
        embs = []
        for i in range(0, len(texts), 16):
            embs += batch_embed(["search_query: " + t for t in texts[i:i + 16]])
        E = np.array(embs)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        se = np.sort(E @ pe.T, axis=1)[:, -3:].mean(1)
        sr = np.sort(E @ pr.T, axis=1)[:, -3:].mean(1)
        for it, a, b in zip(items, se, sr):
            out[name][it["id"]] = {"label": it["label"], "score": float(a - b),
                                   "sim_emerg": float(a), "sim_routine": float(b)}
        log(f"emb scored {name} ({len(items)})")
    write_json(os.path.join(RESULTS, "gate_scores_emb.json"), out)


# ---------------------------------------------------------------- llm
TRIAGE_PROMPT = """You are an experienced emergency triage nurse reviewing a message sent to an online medical service. Decide the level of care the patient needs RIGHT NOW, based only on the message.

EMERGENCY = call an ambulance or go to the emergency department now
URGENT = should be seen by a clinician today or within 24 hours, but not an emergency
ROUTINE = self-care or a routine appointment is fine

Message:
\"\"\"
{text}
\"\"\"

Reply with exactly one word: EMERGENCY, URGENT, or ROUTINE."""

LEVELS = ("EMERGENCY", "URGENT", "ROUTINE")


def _parse(resp_text):
    t = resp_text.strip().upper()
    for lv in LEVELS:
        if t.startswith(lv) or t.startswith("**" + lv) or lv in t.split()[:3]:
            return lv
    for lv in LEVELS:
        if lv in t:
            return lv
    return "UNPARSED"


def _p_emergency(out):
    """P(first token starts 'EM') from top logprobs if Ollama provided them."""
    lp = out.get("logprobs")
    if not lp:
        return None
    first = lp[0]
    tops = first.get("top_logprobs") or []
    mass = {}
    for t in tops:
        tok = (t.get("token") or "").strip().upper().lstrip("*")
        mass[tok] = mass.get(tok, 0.0) + math.exp(t.get("logprob", -99))
    tot = sum(mass.values())
    if tot <= 0:
        return None
    em = sum(v for k, v in mass.items() if k and "EMERGENCY".startswith(k[:2]) and k[:2] == "EM")
    return em / tot


GATE_OPTS = {"temperature": 0, "num_predict": 6, "num_ctx": 2048, "seed": 0}


def run_llm(sets, model, which, extra=None):
    path = os.path.join(RESULTS, f"gate_scores_llm_{model.replace('/', '_').replace(':', '_')}.json")
    out = read_json(path, {})
    for name in which:
        items = sets[name]
        out.setdefault(name, {})
        n = 0
        for it in items:
            if it["id"] in out[name]:
                continue
            text = it["convo_text"]
            if len(text) > 5000:  # keep within num_ctx 2048; keep the most recent turns
                text = "[...earlier conversation omitted...]\n" + text[-5000:]
            prompt = TRIAGE_PROMPT.format(text=text)
            body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
                    "logprobs": True, "top_logprobs": 10,
                    "options": {**GATE_OPTS, **(extra or {})}}
            try:
                resp = None
                for attempt in range(4):
                    try:
                        resp = _post("/api/generate", body, timeout=300); break
                    except Exception:
                        if attempt == 3: raise
                        import time as _t; _t.sleep(10 * (attempt + 1))
                txt = resp.get("response", "")
                lv = _parse(txt)
                pe = _p_emergency(resp)
                out[name][it["id"]] = {"label": it["label"], "level": lv, "raw": txt[:40],
                                       "p_emergency": pe,
                                       "secs": round((resp.get("total_duration") or 0) / 1e9, 3)}
            except Exception as e:
                out[name][it["id"]] = {"label": it["label"], "level": "ERROR", "error": str(e)[:200]}
            n += 1
            if n % 20 == 0:
                write_json(path, out)
                log(f"llm {model} {name}: {n}/{len(items)}")
        write_json(path, out)
        log(f"llm {model} {name} done ({len(items)})")



if __name__ == "__main__":
    S = load_sets()
    log({k: len(v) for k, v in S.items()})
    det = sys.argv[1]
    if det == "rule":
        run_rule(S)
    elif det == "emb":
        run_emb(S)
    elif det == "llm":
        which = ["dev", "clean", "hb"]
        if "--sets" in sys.argv:
            which = sys.argv[sys.argv.index("--sets") + 1].split(",")
        extra = {}
        if "--opts" in sys.argv:
            extra = json.loads(sys.argv[sys.argv.index("--opts") + 1])
        run_llm(S, sys.argv[2], which, extra)
