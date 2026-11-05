# Saying "See a Doctor" Is Not Triage: data, judgments and code

This repository accompanies two papers:

- *Saying "See a Doctor" Is Not Triage: Lexical Safety Rubrics Can Mask Under-Triage in Fine-Tuned Medical
  Language Models* (evaluation paper).
- *OpenHealth: Adding a Triage Safety Gate to a Locally Deployable Medical Question-Answering System* (system paper).

**Nothing here is a medical device. None of the models evaluated here is safe to use for medical advice; the
fine-tuned model in particular under-triages most emergencies.**

## What is included

| Path | What it is |
|---|---|
| `data/clean_benchmark.json` | 96 vignettes written for this study: 72 emergencies (6 in each of 12 specialties) and 24 matched non-emergencies. Fields: `id`, `category`, `question`, `safety_required`, `noisy_query`, `adversarial_query`. |
| `data/orig_cases.json` | The 100-case corpus-adjacent secondary set (44 safety-required). |
| `outputs/` | Every generated answer, per model, case and condition: the main grid, seeds 102/103, the model panel, the ablation (fine-tune without its system prompt, base with it), the public chat-doctor pair, the mitigation runs, HealthBench answers and the gated replies. Retrieved passage text and HealthBench conversation text are removed (see below). |
| `judgments/` | Every per-answer judgment from every judge (Claude Opus 4.8, Claude Opus 5.5, Qwen2.5-7B-Instruct, Mistral-7B-Instruct, the latter also re-run with the full rubric), de-blinded, plus every analysis output (`*_analysis.json`, `review_round*.json`, `stats_B*.json`, `latency_pipeline.json`). |
| `gate/` | The triage-gate detector scores on every development, vignette and HealthBench item, and the scores of the off-the-shelf safety classifiers run on the same items (`guard_scores_*.json`: Llama Guard 3 1B and 8B, ShieldGemma 2B, each with its default policy and with the one-line emergency policy). |
| `code/harness/` | Generation, blinding, judging-batch preparation, aggregation, analysis and figure scripts; `judge_rubric.txt` is the judge prompt; `redflags.py` the red-flag lexicon; `gate.py` the triage prompt and detectors. |
| `code/analysis/` | The earlier analysis scripts that produced the July numbers. |
| `code/training/` | The second training run of the original fine-tuning recipe (`kaggle_replicate_finetune.py`, one Kaggle T4) and `make_scaled.py`, which writes the copies of its adapter at 25-125% strength used in the strength sweep. |
| `docs/R5_safety_eval_survey.md` | The survey of 33 open dialogue fine-tunes and how they evaluate safety. |
| `CLAIMS.md` | A map from every number in both papers to the file and script that produces it. |

## What is not included, and why

- **The 21 retrieval reference documents** are copyrighted guidelines and reviews; the scripts rebuild the index
  from your own copies. Retrieved passage text is removed from the released outputs for the same reason.
- **Training dialogues** (ChatDoctor / HealthCareMagic-100k) are public but contain patient-written text; get them
  from their original release. Statistics about them are given in the paper.
- **HealthBench** conversations are referenced by id; download HealthBench from its official release (MIT licence)
  to reproduce.
- **Model weights** are not redistributed here. The models used are listed in the papers; the commands below fetch
  them.

## Models

Every model runs through Ollama. Pull only the ones a script needs.

```
ollama pull llama3                                   # base model, Llama-3-8B-Instruct, Q4_0
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
# the studied fine-tune: gguf_doctor_model_llama-unsloth.Q4_K_M.gguf, 4,920,734,016 bytes,
# sha256 26126ad1b2762e0c6d104826b1a561d922bf9394cd34ba9a5c6d325540762905
ollama pull nomic-embed-text                         # retrieval embeddings (system paper)
ollama pull llama-guard3:1b; ollama pull llama-guard3:8b; ollama pull shieldgemma:2b   # safety classifiers
```

The deployed fine-tune also carries a model-file system prompt (quoted in the evaluation paper's appendix). To
register it as `openhealth-doctor`, write a Modelfile with `FROM` pointing at the GGUF above and that `SYSTEM`
prompt, then run `ollama create openhealth-doctor -f Modelfile`.

For the strength sweep, run `code/training/kaggle_replicate_finetune.py` on Kaggle (it also converts the adapter to
`openhealth_lora_f16.gguf` with llama.cpp), run `make_scaled.py` next to that file, and
register each copy with a two-line Modelfile (`FROM llama3` and `ADAPTER ./lora_s025.gguf`) as `ohrep-s025`,
`ohrep-s050`, `ohrep-s075`, `ohrep-s100` and `ohrep-s125`.

## Reproducing

Generation ran through Ollama (0.32.5) on a laptop GPU with 4 GB of memory (the July 8B runs used an earlier
Ollama version under Windows; see the papers). Judging used blinded batches of 14 answers with the rubric in
`code/harness/judge_rubric.txt`. Set `OPENHEALTH_ROOT` to the folder that holds `results/` and `data_v2/`.

```
python code/harness/gen_grid.py --models <ollama-model> --tag <tag>     # generate a grid
python code/harness/judge_prep.py <tag> <grid.json>                     # blind into batches of 14
# grade each batch with judge_rubric.txt; save outputs as results/judge_out_<tag>/batch_XX.json
python code/harness/aggregate.py <tag>                                   # de-blind -> judged_<tag>.json
python code/harness/review_stats.py        # main agreement statistics, bootstrap intervals
python code/harness/review_round3.py       # per-seed equivalence, per-judge miss rates, retrieval coverage
python code/harness/abl_analysis.py eval abl   # ablation (also: eval pub for the public chat-doctor pair)
python code/harness/hb_analysis.py         # HealthBench external check
python code/harness/panel_v3.py            # model panel (pooled)
python code/harness/stats_B.py; python code/harness/stats_B2.py   # system paper: generators, gate, end-to-end
python code/harness/figures_v3.py          # Figures 1-2 (figures_v3.py dose: the strength-sweep figure)
# strength sweep, per seed: gen_ablation.py noclarify_ft <ohrep-sXXX> <sXXX> dose_s<seed> with ABL_SEED=<seed>,
# then judge_prep / aggregate as above, then:
python code/harness/dose_analysis.py dose101 gpt6dose101 dose102 gpt6dose102
python code/harness/probe_knows.py; python code/harness/probe_analysis.py      # one-word triage probe
python code/harness/guard_baseline.py run; python code/harness/guard_baseline.py   # safety classifiers vs the gate
```

## Licence

Code: MIT (see `LICENSE`). Benchmark, outputs and judgments: CC BY 4.0. Model weights are not redistributed here.
