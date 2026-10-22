# Claim-to-file map

Every number in both papers, the file that holds it, and the script that regenerates it.
Paths are relative to this release (`judgments/`, `outputs/`, `gate/`, `code/harness/`, `code/analysis/`).
"Primary judge" = Claude Opus 4.8 (July 2026); "newer judge" = Claude Opus 5.5 (October 2026).

## Paper A: "Saying 'See a Doctor' Is Not Triage"

| Claim (section) | Value | File | Script |
|---|---|---|---|
| Escalation, pooled, primary judge (4.1, abstract) | base 69/288 = 0.240, fine-tune 69/288 = 0.240 | judgments/judged_clean_s101.json, judgments/analysis_clean_s101.json | code/analysis/full_analysis.py clean_s101 |
| Per-condition escalation (Table 2) | 0.29/0.21/0.24/0.22 vs 0.24/0.28/0.19/0.25 | same | same |
| Clean condition received passages for 38/72 emergencies; 13/38->8/38 (base), 9/38->10/38 (FT) | | judgments/review_round3.json (retrieval_coverage) | code/harness/review_round3.py |
| Case-level TOST per seed and pooled (4.1) | seed 101 +/-6.3; seed 102 -3.5 [-10.1, 3.1]; pooled -1.9 [-6.4, 2.6] | judgments/review_round3.json (tost_90ci_ft_minus_base) | code/harness/review_round3.py |
| No-retrieval deficit pooled over seeds (4.1) | 9.5 points [0.7, 18.8], sign-flip p=0.05 | judgments/review_round3.json (noretrieval_deficit_pooled_seeds_95) | same |
| No-retrieval escalation per seed (4.1) | base 0.29/0.33/0.41, FT 0.24/0.21/0.29 | judgments/review_stats.json (per_seed) | code/harness/review_stats.py |
| Keyword failure 65% vs 10%; 187/28 = 6.7x (4.2) | kw-safe 0.351 / 0.903 | judgments/judged_clean_s101.json | code/harness/figures_v3.py (fig1 stats) |
| Urgency-only list (4.2) | safe 0.27/0.35, raw 0.89/0.81, FDR 0.27/0.43 | judgments/robustness_stats.json | code/analysis/robustness_stats.py |
| Table 3 agreement + bootstrap CIs (4.3) | kappa 0.61 [0.49,0.71] / 0.03 [-0.01,0.06]; raw, FDR, AUROC | judgments/review_stats.json (bootstrap_pooled_288) | code/harness/review_stats.py |
| No-retrieval agreement (4.3) | raw 0.85->0.35, kappa 0.67->0.07 | judgments/review_stats.json (bootstrap_noretrieval_72) | same |
| Keyword share in under-triage answers (4.3, Fig 1b, abstract) | 0.18 vs 0.89 | judgments/review_stats.json (kw_share_in_undertriage) | same |
| Distinct keywords, under-triage vs escalating (4.3) | FT 1.57 vs 2.41; base 0.21 vs 2.41 | judgments/judged_clean_s101.json | code/harness/figures_v3.py (fig1 stats) |
| Best-threshold kappa; word-boundary variant (4.3) | 0.75 vs 0.32; 0.67 vs 0.04 | judgments/review_stats.json | code/harness/review_stats.py |
| Failure modes (Table 4) | base 41/51 questions only; FT 26/55 wrong assessment, 14 vague; rubric keyword 10/51 vs 47/55 | judgments/review_round2.json (failure_modes), judgments/review_round3.json (failure_mode_rows_contain_rubric_keyword) | failure-mode coding batches; code/harness/review_round3.py |
| Per-judge Table 5: esc, raw, kappa, kw given under-triage, McNemar | see table | judgments/per_judge_v2.json, judgments/review_round3.json (kw_safe_given_under_triage) | code/harness/per_judge_v2.py, review_round3.py |
| Open-judge face validity (4.5) | Qwen 32/41, Mistral 6/41, newer Claude 0/41 | judgments/review_round3.json (open_judge_face_validity) | code/harness/review_round3.py |
| Mistral re-run with the full Claude rubric (4.5) | base 0.75 vs FT 0.88 esc; kappa 0.23 / 0.40; questions-only credited 25/41; kw given under-triage 0.17 vs 0.56 | judgments/judge_local_full_s101.json, judgments/judge_local_full_analysis.json | code/harness/judge_local_full.py |
| GPT-6 judge (4.5, Table 5): base esc 0.26, kappa 0.49; FT esc 0.19, kappa 0.06, FDR 0.78, kw given under-triage 0.86; agreement with newer Claude judge 97.2% (kappa 0.92), with primary 91.0% (kappa 0.76); question-only credited 0/41 | | judgments/judged_gpt6none.json, judgments/gpt6_judge_analysis.json | code/harness/codex_judge.py, gpt6_analysis.py |
| Newer-judge test-retest (3, App. G) | 190/192 = 99.0%, kappa 0.98 | judgments/review_round3.json (opus55_test_retest) | same |
| Two binarisations (4.5) | lenient pooled 0.36->0.07; no-retrieval 0.26->0.21 (primary), 0.68->0.31 (newer) | judgments/format_robustness.json | code/harness/format_robustness.py |
| Newer judge drift (4.5) | 92.4%, kappa 0.79 | judgments/review_stats.json, judgments/calib_analysis.json | review_stats.py, calib_analysis.py |
| Blinding (Limitations) | FT names "Chat Doctor" 73/384, greeting 384/384 (base 0, 104/383); escalation 16/63 vs 53/225 | judgments/review_round3.json (blinding) | code/harness/review_round3.py |
| Corpus register (4.6) | 79.5%; doctor 69.3%, consult 27.2%, 911 0.0% | judgments/corpus_register.json | (July corpus scan) |
| Corpus audit, newer judge (4.6) | 114/510 emergencies; 17/114 escalated | judgments/review_round2.json (corpus_audit_v3) | audit batches (uid only) |
| FT keyword rate on controls (4.6) | 74/96 | judgments/review_round3.json (kw_rate_controls) | code/harness/review_round3.py |
| Mitigation with controls (4.7) | base 22%->97% (140/144), controls 11/48; FT 15%->42%, controls 45/48 | judgments/judged_mitig.json, judgments/mitig_analysis.json | code/harness/prep_mitig.py, mitig_analysis.py |
| Ablation and public chat-doctor pair (4.8) | FT no-sys kappa 0.11, FDR 0.69; base+FT prompt kappa 0.65; FT current runtime kappa 0.05; base Q4_K_M kappa 0.73; public FT kappa 0.05, kw-safe 0.86, esc 22/72 vs 16/72 (p 0.34) | judgments/abl_analysis.json, judgments/pub_analysis.json | code/harness/gen_ablation.py, abl_analysis.py |
| Panel, single judge, no retrieval (Table 6, Fig 2) | see table | judgments/review_round2.json (panel_noretrieval_opus55) | code/harness/figures_v3.py |
| Panel, pooled, case-clustered (4.9, App. E) | Medllama2 +34, MedGemma 1 +20 [12, 29], MedGemma 1.5 +8 [0.7, 15.3] p=0.04 | judgments/review_round3.json (panel_case_clustered_4cond_95), judgments/panel_v3.json | code/harness/review_round3.py, panel_v3.py |
| HealthBench external check (4.10) | esc 61/139 vs 51/139 (p 0.17); kappa 0.62 [0.50,0.74] vs 0.11 [0.03,0.21]; kw given under-triage 25/78 vs 70/88 | judgments/judged_hb3.json, judgments/hb_analysis_v3.json | code/harness/gen_hb.py, hb_analysis.py |
| MCQ (Discussion) | MedQA 0.557 vs 0.542 (p 0.72); PubMedQA 0.650 vs 0.688 (p 0.29), unpaired | judgments/mcq_results.json, judgments/review_stats.json (mcq) | code/harness/review_stats.py |

## Paper B: "OpenHealth ... with a Triage Safety Gate"

| Claim | File | Script |
|---|---|---|
| ROUGE table; 25.5 s vs 60.0 s; 84 vs 219 words | rouge_eval_results.json (original project) | inline recomputation (see v3/LEDGER.md) |
| Generator triage table, paired tests, retrieval subset, test-retest | judgments/stats_B.json | code/harness/stats_B.py |
| Gate detection table | gate/gate_analysis.json, gate/gate_selection.json | code/harness/gate_analysis.py, gate_select.py |
| Rules' marginal value, error analysis, label-only gate end-to-end, McNemar contrasts | judgments/stats_B2.json | code/harness/stats_B2.py |
| End-to-end table (strict outcome) and HealthBench end-to-end | judgments/stats_B2.json (strict_endtoend, healthbench_mg15) | code/harness/stats_B2.py |
| Mixed plain+gated re-judge (scoring of contradicted instructions) | judgments/judged_mixed.json, judgments/mixed_rejudge.json | judge_prep.py mixed + grid_judge_gated.wf.js; inline analysis (v3/LEDGER.md) |
| 8B serving speed | judgments/latency_ft.json | code/harness/latency_8b.py |
| Per-message latency of the deployed path (single 2.9 s; split 21.6 s; CPU gate 9.6 s) | judgments/latency_pipeline.json | code/harness/latency_pipeline.py |
| Gate-conditioned instruction on HealthBench (strict 119/139; non-emergent 121/134; contradiction 7/128) | judgments/judged_hbgi.json, judgments/hb_gateinstr_analysis.json | code/harness/gen_hb_instr.py, hb_gateinstr_analysis.py |
