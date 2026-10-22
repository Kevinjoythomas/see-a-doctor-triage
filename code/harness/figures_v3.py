"""Paper A figures (v3). Vector PDF + PNG into v3/papers/A/figs/.

fig1: the illusion. (a) keyword-"safe" rate vs blinded-judge escalation, base vs fine-tune
      (72 emergencies x 4 conditions = 288 answers/model, judged_clean_s101.json);
      (b) mean keyword count in under-triaging vs escalating answers (operating-point shift).
fig2: model pairs (base -> medical model): escalation and keyword<->judge kappa per pair,
      read from v3/results/panel_v3.json (written by panel_v3.py).
Palette: categorical slots 1-2 (#2a78d6 blue = base, #eb6834 orange = fine-tune / medical),
validated with the dataviz validator (light surface #fcfcfb); values are direct-labelled.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, RESULTS_V2, SAFETY_KW, read_json

OUT = "./v3/papers/A/figs"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def bar_labels(ax, bars, fmt="{:.0%}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, fmt.format(h), ha="center", va="bottom",
                fontsize=8, color=INK)


def fig1():
    rows = read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))
    em = [r for r in rows if r["safety_required"]]
    esc = lambda r: r["appropriate_escalation"] in (True, "true", 1)
    kwc = lambda r: sum(k in r["answer"].lower() for k in SAFETY_KW)
    stats = {}
    for m in ("llama3", "openhealth-doctor"):
        s = [r for r in em if r["model"] == m]
        stats[m] = {"kw_safe": np.mean([kwc(r) > 0 for r in s]), "esc": np.mean([esc(r) for r in s]),
                    "kw_under": np.mean([kwc(r) for r in s if not esc(r)]),
                    "kw_esc": np.mean([kwc(r) for r in s if esc(r)]), "n": len(s),
                    "share_under": np.mean([kwc(r) > 0 for r in s if not esc(r)]),
                    "share_esc": np.mean([kwc(r) > 0 for r in s if esc(r)])}
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(2); w = 0.34
    for i, (m, col, lab) in enumerate((("llama3", BLUE, "Base (Llama-3-8B)"),
                                        ("openhealth-doctor", ORANGE, "Fine-tune (ChatDoctor QLoRA)"))):
        bars = a.bar(x + (i - 0.5) * (w + 0.02), [stats[m]["kw_safe"], stats[m]["esc"]], w, color=col,
                     label=lab, edgecolor="white", linewidth=1)
        bar_labels(a, bars)
    a.set_xticks(x); a.set_xticklabels(['keyword rubric:\n"safe"', "blinded judge:\nescalates"])
    a.set_ylim(0, 1.08); a.set_ylabel("share of 288 emergency answers")
    a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.set_title("(a) The rubric sees a safety gain the judge does not", fontsize=9, loc="left", color=INK)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    for i, (m, col) in enumerate((("llama3", BLUE), ("openhealth-doctor", ORANGE))):
        bars = b.bar(x + (i - 0.5) * (w + 0.02), [stats[m]["share_under"], stats[m]["share_esc"]], w, color=col,
                     edgecolor="white", linewidth=1)
        bar_labels(b, bars)
    b.set_xticks(x); b.set_xticklabels(["under-triaging\nanswers", "escalating\nanswers"])
    b.set_ylabel("share with at least one rubric keyword"); b.set_ylim(0, 1.08)
    b.yaxis.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True)
    b.set_title("(b) Fine-tuning puts keywords into answers that do not escalate", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig1_illusion")
    return stats


def fig2():
    rr = read_json(os.path.join(RESULTS, "review_round2.json"))
    if not rr:
        print("review_round2.json missing; fig2 skipped"); return
    P = rr["panel_noretrieval_opus55"]
    G3 = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"; M1 = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"
    M15 = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
    pairs = [("Llama-3 →\nChatDoctor", "llama3", "openhealth-doctor"), ("Llama-2 →\nMedllama2", "llama2", "medllama2"),
             ("Gemma 3 →\nMedGemma 1", G3, M1), ("Gemma 3 →\nMedGemma 1.5", G3, M15)]
    pub = read_json(os.path.join(RESULTS, "pub_analysis.json"), {}) or {}
    for k in ("llama3-q4km", "public-chat-doctor"):  # public chat-doctor pair, same judge, no retrieval
        if k in pub:
            P[k] = {"esc": pub[k]["escalation"], "esc_ci": pub[k]["escalation_ci"], "kappa": pub[k]["kappa"]}
    pairs.insert(1, ("Llama-3 → public\nchat-doctor", "llama3-q4km", "public-chat-doctor"))
    pairs = [p for p in pairs if p[1] in P and p[2] in P]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    xs = np.arange(len(pairs)); w = 0.34
    for ax, key, title, ylab in ((a, "esc", "(a) Judged emergency escalation (95% CI)", "escalation rate"),
                                 (b, "kappa", "(b) Keyword rubric vs judge agreement", "Cohen's κ")):
        for side, idx, col, lab in ((-1, 1, BLUE, "base model"), (1, 2, ORANGE, "medical model")):
            vals = [P[p[idx]][key] for p in pairs]
            bars = ax.bar(xs + side * (w + 0.02) / 2, vals, w, color=col, label=lab, edgecolor="white", linewidth=1)
            if key == "esc":
                lo = [v - P[p[idx]]["esc_ci"][0] for v, p in zip(vals, pairs)]
                hi = [P[p[idx]]["esc_ci"][1] - v for v, p in zip(vals, pairs)]
                ax.errorbar(xs + side * (w + 0.02) / 2, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
                for bx, v, p in zip(bars, vals, pairs):
                    ax.text(bx.get_x() + bx.get_width() / 2, P[p[idx]]["esc_ci"][1] + 0.015, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7, color=INK)
            else:
                bar_labels(ax, bars, "{:.2f}")
        ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in pairs], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left", color=INK); ax.set_ylabel(ylab)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylim(0, 1.08)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_pairs")


if __name__ == "__main__":
    s = fig1()
    print({m: {k: round(float(v), 3) for k, v in d.items()} for m, d in s.items()})
    fig2()
