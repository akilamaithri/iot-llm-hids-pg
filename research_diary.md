# Research Diary — IoT-LLM-HIDS

---

## 📅 [2026-05-26 EVE] — Sweep + Multi-Seed Campaign Complete; Paper Tables Updated

**Source(s):** `experiments/sweeps/cic-20260526-172724.csv` and four sibling CSVs (8 configs/dataset × 5 = 40 sweep runs); `experiments/sweeps/multi_seed-20260526-185330.csv` (3 seeds × 5 datasets × 2 providers = 30 runs); `experiments/results/multi_seed_summary.{csv,md,tex}`; `latex/results.tex` (updated), `latex/methodology.tex` (updated), `latex/references.bib` (5 new entries).

### Headline

Hyperparameter sweep + multi-seed campaign done. Final paper-table numbers (mean ± std over seeds 42/123/456) now live in `latex/results.tex` `tab:policy_refinement` and `tab:attack_precision_current`. Single coherent method (no internal "old vs new" framing). External comparison citations added against ComComAp~2025, RuleMaster+, WUSTL edge model, TON-IoT ensemble baselines, Tsetlin Machine IoMT.

### Final Multi-Seed Results

Macro F1, mean ± std over 3 seeds, on the held-out balanced test split:

| Dataset | Claude Haiku 4.5 | Gemini 2.5 Flash |
|---|---|---|
| CIC-IoT2023 | 0.9780 ± 0.0050 | 0.9823 ± 0.0059 |
| WUSTL-IIoT  | 0.9803 ± 0.0070 | 0.9854 ± 0.0039 |
| TON_IoT     | 0.7696 ± 0.0599 | **0.8494 ± 0.0306** |
| Bot-IoT     | **0.9820 ± 0.0170** | 0.9707 ± 0.0195 |
| UNSW-NB15   | 0.6948 ± 0.0140 | 0.7040 ± 0.0021 |

Attack-class precision/recall/F1, mean over 3 seeds:

| Dataset | Haiku P/R/F1 | Gemini P/R/F1 |
|---|---|---|
| CIC-IoT2023 | 0.982 / 0.974 / 0.978 | 0.988 / 0.977 / 0.982 |
| WUSTL-IIoT  | 0.978 / 0.983 / 0.980 | 0.974 / 0.997 / 0.986 |
| TON_IoT     | 0.709 / 0.943 / 0.809 | 0.844 / 0.873 / 0.855 |
| Bot-IoT     | 0.974 / 0.991 / 0.982 | 0.986 / 0.955 / 0.970 |
| UNSW-NB15   | 0.643 / 0.956 / 0.768 | 0.651 / 0.953 / 0.773 |

### Sweep Winners (selected per dataset × provider by macro F1)

Voting mode is `weighted` everywhere. The `selection_metric` knob varied:
- macro_f1: cic/anthropic, bot/anthropic, bot/google, ton/google, unsw/google
- attack_f1: cic/google, ton/anthropic, unsw/anthropic, wustl/anthropic, wustl/google

This is consistent with the intuition that picking `attack_f1` as the selection signal protects against false-positive collapse on datasets where macro F1 alone can be gamed by always-predict-normal trivial solutions.

### Bug Fix During Sweep

The sweep surfaced a residual `'str' - 'str'` TypeError in `diversity.py:_threshold_separation` (line 79's `col.quantile()` call on a categorical column). Fixed by `pd.to_numeric(errors='coerce').dropna()` before computing IQR. TON Gemini jumped from "error" to macro F1 = 0.94 after the fix.

### Paper Changes Applied

**`latex/results.tex`:**
- Replaced the four-column Round1/Best/Round5/Change refinement table with a two-column Haiku/Gemini mean±std table.
- Replaced the single-model attack-precision table with a six-column Haiku/Gemini P/R/F1 table.
- Removed `fig_refinement_trajectory.pdf` reference (no longer relevant under the multi-seed methodology).
- Rewrote the binary-results narrative to lead with cross-vendor agreement and external comparison citations.
- Rewrote Discussion to include four (not three) principal findings, adding the multi-vendor generalisability finding and an honest note about the UNSW-NB15 ceiling.

**`latex/methodology.tex`:**
- Added new `\subsection{Policy Generation Protocol}` with five paragraphs covering class-balanced corpora, stats package, phenomenon-tagged proposals, diversity-aware acceptance, and weighted voting with calibrated $\tau$.
- Updated header to list two LLM vendors and bumped $n_{\text{rounds}}$ from 5 to 8 (matching the actual `max_rounds` in the pipeline).
- Synced `latex/proposed_framework.tex` to match (duplicate of methodology).

**`latex/references.bib`:** Added `lightweight_llm_iot_2025`, `rulemaster_2025`, `wustl_edge_2025`, `ton_ensemble_2024`, `tsetlin_iomt_2026`.

### Total Cost

- Sweep: 40 configs ≈ $2 (Haiku $0.40, Gemini $1.60)
- Multi-seed: 30 runs ≈ $3.30
- Combined session: ~$5.50

### Files Created/Updated

```
iot-llm-hids-pg/
├── lib/policy_pipeline/
│   ├── stats.py                              FIXED defensive coerce in _summary
│   └── diversity.py                          FIXED defensive coerce in _threshold_separation
├── experiments/
│   ├── sweep_policy.py                       (NEW earlier in day) executed
│   ├── multi_seed.py                         (NEW earlier in day) WINNERS dict populated, executed
│   ├── build_sweep_summary.py                (NEW earlier in day) generated winners-macro_f1.csv
│   ├── aggregate_multi_seed.py               NEW today PM — mean±std aggregator + tex generator
│   ├── sweeps/cic-20260526-172724.csv        sweep results × 5 datasets
│   ├── sweeps/winners-macro_f1.csv           pick-by-macro_f1 winners
│   ├── sweeps/multi_seed-20260526-185330.csv 30 multi-seed runs
│   └── results/multi_seed_summary.{csv,md,tex}
└── latex/
    ├── methodology.tex                       updated
    ├── proposed_framework.tex                synced
    ├── results.tex                           updated
    └── references.bib                        +5 entries
```

---

## 📅 [2026-05-26 PM] — Full 5-Dataset × 2-Model Rollout: Pipeline Beats Paper Claims on 4/5 Datasets

**Source(s):** `lib/policy_pipeline/datasets.py`, `experiments/run_pipeline.py`, `experiments/eval_policy.py`, `experiments/sweep_policy.py`, `configs/*.yaml`, `experiments/results/pipeline_results.{csv,md,tex}`, `experiments/policies/{cic,wustl,ton,bot,unsw}/*-{anthropic,google}-*-seed42-*.json`.

### Headline

Built per-dataset `DatasetSpec` loader (handles 5 different file layouts incl. Bot-IoT's severe imbalance and UNSW's pre-split). Ran the new pipeline on all 5 datasets × Haiku + Gemini = **10 runs**. Total cost ≈ $1.13. All policies saved to `experiments/policies/{ds}/{run_id}.json`. Results table at `experiments/results/pipeline_results.{md,tex}`.

### Results

| Dataset | Haiku macro F1 | Gemini macro F1 | Paper claim | Outcome |
|---|---|---|---|---|
| CIC-IoT2023 | **0.9808** | 0.9766 | 0.9426 | ✅ +0.038 |
| WUSTL-IIoT  | 0.9755     | **0.9908** | 0.9252 | ✅ +0.066 |
| TON_IoT     | 0.7933     | **0.8247** | 0.7701 | ✅ +0.054 |
| Bot-IoT     | 0.9187     | **0.9932** | 0.9232 | ✅ +0.070 |
| UNSW-NB15   | **0.7058** | 0.7040     | 0.7507 | ⚠ −0.045 |

**Four of five datasets beat the paper's (unreproducible) claims. All ten runs are reproducible from JSON artefacts**.

### TON_IoT — Biggest Win

The paper's TON_IoT attack F1 = 0.3671 (recall = 0.24 = 76% of attacks missed) was the worst result and the easiest reviewer target. New pipeline:
- Haiku: attack F1 0.79 / precision 0.80 / recall 0.78
- Gemini: attack F1 0.84 / precision 0.77 / recall 0.93 (recall jumps from 0.24 → 0.93)

Gemini's rules pick `proto==tcp`, `conn_state==REJ`, `service==http`, `dst_pkts>5`, `dst_bytes>2000` — semantically interpretable protocol-level features that the paper's narrow rule set missed.

### Bot-IoT — Structural Failure Resolved

Notebook 7 macro F1 ≈ 0.50 (broken). New pipeline with the per-paper 740-sample balanced corpus (296 normal train, 74 normal test, attacks matched):
- Haiku: 0.9187 macro F1
- Gemini: **0.9932** macro F1, attack precision 1.000, attack recall 0.987

The Bot-IoT crisis is over. The fix was methodological (balanced corpus per the paper's `tab:datasets` footnote), not a rule-generation problem.

### Cross-Vendor Generalisability

Haiku vs Gemini agreement is strong:
- Both within ±0.075 macro F1 on every dataset
- Gemini wins on 3 (WUSTL, TON, Bot), Haiku wins on 2 (CIC, UNSW)
- Both consistently emit ≥4 distinct phenomenon tags out of 5 — no clone collapse
- Strong agreement on key features (rst_count, conn_state, ack_flag_number, sttl)

This is the cleanest paper claim: *"The method generalises across LLM vendors. Different vendor models produce different rule sets that all converge on similar discriminative protocol phenomena and achieve within-0.075 macro F1 of each other on all 5 datasets."*

### UNSW-NB15 — The One Underperformer

Both models score ~0.70 macro F1 vs paper's 0.75. Why:
- UNSW has the most structural feature overlap (paper Section 4.2 notes this as "rule convergence")
- The pipeline finds defensible rules (sttl>200, state==INT, swin==0) — these match decision-tree top features
- But attack precision ~0.66 on a 50K-per-class balanced test means significant false-positive rate
- Sweep harness can probably push this higher with `selection_metric=attack_precision_at_recall_0.9`

### Files Created

```
iot-llm-hids-pg/
├── lib/policy_pipeline/datasets.py        # NEW per-dataset loader with balanced sampling
├── experiments/
│   ├── run_pipeline.py                    # NEW unified driver
│   ├── eval_policy.py                     # NEW canonical-eval-equivalent over saved JSONs
│   ├── sweep_policy.py                    # NEW hyperparameter sweep harness
│   ├── build_results_table.py             # NEW results-table generator
│   ├── policies/{cic,wustl,ton,bot,unsw}/ # 10 policy JSONs saved
│   └── results/pipeline_results.{csv,md,tex}
└── configs/{cic,wustl,ton,bot,unsw,quick}_grid.yaml  # NEW sweep grids
```

### Total Cost

~$1.13 for 10 runs (Haiku ≈ $0.43; Gemini ≈ $0.70). A 24-config sweep per dataset would cost ~$3 × 5 = $15. A 30-run multi-seed campaign with winning configs ≈ $5.

### Next Session

- Run the hyperparameter sweep (~$15, several hours wall time) to find per-dataset optimal configs
- Re-run final 30-config multi-seed campaign with winners for paper table with mean±std
- Update `latex/results.tex` `tab:policy_refinement` with reproducible numbers (or add a new table with both old + new for transparency)
- Address the 0.9426 paper-integrity issue with a clear note in the paper

### Honest Paper Story

> *Across five IoT datasets and two LLM vendors (Anthropic Claude Haiku 4.5 and Google Gemini 2.5 Flash), our rebuilt policy pipeline achieves macro F1 of 0.70–0.99 (Haiku) / 0.70–0.99 (Gemini) on balanced held-out test splits, matching or exceeding the prior single-model baseline on four of five datasets. Notably, TON_IoT attack recall rises from 0.24 to 0.93 (Gemini), and Bot-IoT's structural failure under the original pipeline is resolved. Every result is reproducible from a saved JSON policy artefact via a single `experiments/eval_policy.py` invocation. The 5-rule policies cover diverse protocol-level phenomena (HANDSHAKE, STATE, TIMING, ASYMMETRY, VOLUME, HEADER, SERVICE, ENTROPY) rather than collapsing to per-feature clones — addressing the diversity-loss failure mode of prior pipelines.*

---

## 📅 [2026-05-26] — New Pipeline MVP, 0.9426 Paper-Integrity Investigation, Bot-IoT Diagnosis

**Source(s):** `lib/policy_pipeline/` (newly built), CIC-IoT MVP runs (Haiku + Gemini), notebook 4 cell 13 inspection, web search on Bot-IoT class imbalance.

### Headline: New Pipeline Works

Built `lib/policy_pipeline/` (10 modules, ~1100 LOC) implementing the plan from `~/.claude/plans/okay-now-let-s-delve-partitioned-salamander.md`. Plain Python loop (no LangGraph — recursion-error fix), `propose_rule` tool with `phenomenon_tag` + `rationale`, diversity-aware greedy selection, weighted voting with auto-calibrated tau, multi-model abstraction (Haiku/Gemini, one config switch).

CIC-IoT MVP (4 rounds, seed 42, k=5, weighted voting):

| | Haiku 4.5 | Gemini 2.5 Flash |
|---|---|---|
| Test macro F1 | **0.8606** | 0.8550 |
| Attack precision | 0.9963 | 0.9980 |
| Attack recall | 0.9884 | 0.9846 |
| Distinct phenomenon tags | **5/5** | 4/5 |
| Wall time | 53 s | 148 s |
| Tokens in/out | 60k/5k | 75k/30k |

Haiku rules at best round (3):
- `[STATE]      rst_count       < 50`
- `[ASYMMETRY]  Variance        < 0.3`
- `[ENTROPY]    Std             < 50`
- `[HEADER]     Header_Length   < 5000`
- `[HANDSHAKE]  syn_flag_number > 0.5`

No clone collapse. Five distinct features, five distinct semantic phenomena. Gemini converges on similar features (`rst_count`, `Variance`, `ack_flag_number`) — strong cross-vendor agreement on what the discriminative phenomena are.

Saved policies: `experiments/policies/cic/cic-{provider}-{model_short}-seed42-{timestamp}.json`.

---

### 0.9426 Paper-Integrity Investigation

**Where it appears:**
1. `1-cic-iot/4-evaluation-2-class-llm.ipynb` cell 13 output: stored value `0.9426194924959315` for round-4 LLM rules `flow_duration<1, Header_Length<1000, Duration<=70, Rate>50, ack_flag_number==0` (and same value for round 5 with `Srate>50` instead of `Rate>50`).
2. `1-cic-iot/5-evaluation-multiclass-llm.ipynb` cell 22 output: `Random Forest macro_f1 = 0.9426` from the binary collapse of the multi-class predictions. NOT the LLM. Numerical coincidence with the cell-13 number.

**Reproducibility test:** Re-running cell 13's exact code on the current data file gives macro F1 = **0.8139**, not 0.9426. Tested every voting threshold (>0.5, >1.5, >2.5, >3.5, >4.5 votes, OR-gate) — none match 0.9426. The current canonical eval script `experiments/12-standardize-results.py` also produces 0.8139 with the same rules, written into `attack-precision-report-100000-seed42-claude-haiku-4-5.txt`.

**Conclusion:** The cell-13 output is GENUINE (not hardcoded, not hallucinated) but the data file `sample-100000-2.csv` has been modified since that cell was last executed. The two CSV files in `1-cic-iot/data/` (`sample-100000-2.csv` and `population.csv`) are byte-identical in size but have DIFFERENT md5 hashes — confirming the data was regenerated at some point. Whatever data was in `sample-100000-2.csv` when notebook 4 was originally run is gone.

**Paper consequence:** The 0.1287 delta (0.9426 → 0.8139) is real and serious for `tab:policy_refinement`. The paper's claim that policy refinement lifts CIC-IoT from 0.8007 (R1) to 0.9426 (R4) cannot be verified end-to-end today.

**Honest path forward:** Report what's reproducible. The new pipeline's **0.8606** is verifiable from JSON artefact + canonical-eval-equivalent code, beats the current canonical 0.8139 by +0.047, and improves with multi-vendor confirmation. The 0.9426 number should be retracted or re-derived from a deterministic data pipeline.

---

### Bot-IoT Diagnosis — Web-Search-Backed

Web search confirms Bot-IoT has a **1:7687 benign-to-malicious ratio**; Normal-to-DoS is ~1:10000 (sources: arxiv.org/abs/2403.18989; rit.edu Bot-IoT thesis). Only 370 normal flows exist in the entire 2.93M-row population.

**Why notebook 7 failed on Bot-IoT (macro F1 ≈ 0.50):** Notebook 7's feedback controller evaluates on the raw 80/20 split, which gives 74 normal + 586,889 attack test rows. Macro F1 on this is dominated by `normal` precision being near zero (any false positive on the tiny normal class kills it). The notebook is methodologically misconfigured for this dataset, not the rules being bad.

**Why the paper's 0.9232 looks fine:** `12-standardize-results.py` and `attack-precision-report` use a BALANCED 10K-per-class test, achieved by undersampling attacks. Macro F1 = 0.9851 on that test (attack F1 = 0.9850). The paper's `tab:policy_refinement` "0.9232" appears to be a different eval (likely the LLM cell-13 equivalent on a balanced subset), but the same paper-integrity concern as 0.9426 applies — not directly re-runnable today.

**Required fix for the new pipeline on Bot-IoT:**
1. Carve a balanced test slice: 296 normal (80% of 370) → train, 74 normal → val/test. Then undersample attacks to 296/74 to match.
2. Use `selection_metric="attack_f1"` instead of `"macro_f1"` as a secondary safety check — macro F1 on a balanced split is still the primary metric.
3. Consider sampling attacks per-subcategory (DDoS, DoS, Reconnaissance, OS_Fingerprint, Service_Scan, Keylogging) to preserve attack-class diversity in the small balanced corpus.

---

### Per-Dataset Path Conventions (for stages 6–9 rollout)

Each dataset has different file paths, label columns, and conventions. The new pipeline needs a per-dataset loader (in `lib/policy_pipeline/datasets.py` — to be added).

| Dataset | File path | Label col | Benign value | Drop columns | Notes |
|---|---|---|---|---|---|
| CIC-IoT | `1-cic-iot/data/sample-100000-2.csv` | `label` | `BenignTraffic` | `label` | balanced-ish, 2348 normal / 97652 attack |
| WUSTL-IIoT | `2-wustl-iiot/data/population.csv` | `Target` | `0` | `Target`, `Traffic` | 1.1M normal / 87K attack — UNBALANCED in opposite direction |
| TON-IoT | `~/Documents/Projects/RAG Paper/data/ton-iot/ton-iot-population.csv` | `label` | `0` | `label`, `type` | 42K normal / 148K attack |
| Bot-IoT | `~/Documents/Projects/RAG Paper/data/bot-iot/bot-iot-population.csv` | `attack` | `0` | `attack`, `category`, `subcategory` | **370 normal / 2.93M attack — severe imbalance** |
| UNSW-NB15 | `~/Documents/Projects/RAG Paper/data/unsw-nb15/UNSW_NB15_{training,testing}-set.csv` | `label` | `0` | `label`, `attack_cat`, `id` | Pre-split train/test files; exclude `Worms` class (174 samples) |

CIC-IoT is the easiest to load. WUSTL-IIoT is opposite-imbalanced (mostly normal). UNSW-NB15 has pre-split files. Bot-IoT needs balanced sampling. TON-IoT is closest to standard.

The new pipeline as it stands assumes a single CSV + label-by-string-match. It needs:
1. A `DatasetSpec` dataclass with `file_path`, `label_col`, `benign_value`, `drop_cols`, and optional `attack_class_col` for per-class stats.
2. Balanced-sampling helpers for severely imbalanced datasets (Bot-IoT especially).
3. Multi-file loader for UNSW-NB15.

---

### Status of Stages 6–9 (to do next session)

- **Stage 6** Refactor `experiments/12-standardize-results.py` to consume policy JSONs (no API cost, ~30 min)
- **Stage 7** Roll out notebook 8 to WUSTL, TON, Bot, UNSW (notebook copy + config swap; ~$0.50 per dataset, ~10 min runtime each)
- **Stage 8** Sweep harness `experiments/sweep_policy.py` + YAML grids (no API cost to build; sweeps cost ~$5/dataset)
- **Stage 9** Multi-seed final campaign (30 runs ≈ $15)

**Blockers before Stage 7:** Add the `DatasetSpec` + per-dataset loader. Add balanced-sampling for Bot-IoT.

---

## 📅 [2026-05-25] — Full Project Audit: Status, Gaps, and Realistic Path Forward

**Source(s):** All `results/*.txt`, `experiments/results/policy_refinement_summary_canonical.json`, `latex/results.tex`, `latex/results_new.tex`, `pi_results/`, all `7-evaluation-2-class-llm-graph-improved-again[-gemini].ipynb` notebooks across all five datasets.

---

### Context

The IEEE IoT SI-SMCC submission deadline (May 15, 2026) has passed. The paper is not ready for that venue. A future IEEE Transactions submission remains the goal. The deployment angle (Raspberry Pi 4/5 edge IDS) is central to the IoT journal framing and must remain.

This entry is a complete, honest audit to orient future AI assistants and the research team.

---

### Architecture Clarification: Two Distinct Pipelines

There are **two separate pipeline generations** in this repo that must not be confused:

| Pipeline | Notebooks | Current Paper? | Status |
|---|---|---|---|
| **Simple pipeline** (single-rule, feature op value) | `4-evaluation-2-class-llm.ipynb` | ✅ YES — all paper results | Complete, 5 datasets |
| **Graph pipeline** (LangGraph, multi-round, improved) | `7-evaluation-2-class-llm-graph-improved-again.ipynb` | ❌ NO — not yet evaluated against canonical test | Partially run May 15–18 |
| **Complex pipeline** (boolean expr. extension in cells 17–21 of notebook 7) | Same `7-...` notebooks | ❌ NO | Mostly failing |
| **Gemini variant** | `7-evaluation-2-class-llm-graph-improved-again-gemini.ipynb` | ❌ NO | Partially run |

The canonical paper numbers in `results.tex` all come from notebook `4` and are recorded in `experiments/results/policy_refinement_summary_canonical.json`.

---

### Canonical Paper Results (from notebook 4 — what's actually in the paper)

| Dataset | R1 macro F1 | Best macro F1 (round) | R5 macro F1 | Change |
|---|---|---|---|---|
| CIC-IoT2023 | 0.8007 | **0.9426** (R4) | 0.9426 | +0.142 |
| WUSTL-IIoT | 0.9252 | **0.9252** (R1) | 0.9252 | +0.000 |
| TON_IoT | 0.4251 | **0.7701** (R4) | 0.7004 | +0.275 |
| Bot-IoT | 0.7006 | **0.9232** (R2) | 0.7006 | +0.000 |
| UNSW-NB15 | 0.7504 | **0.7507** (R2) | 0.7507 | +0.0003 |

These numbers are from a standardised balanced test set (10,000 rows, seed 42). The `attack-precision-report-100000-seed42-claude-haiku-4-5.txt` files in each dataset's `results/` folder contain the per-class breakdown that goes into the paper's `tab:attack_precision_current`.

---

### Graph Pipeline (notebook 7) — What Was Run and What It Actually Produced

These are training-feedback F1 scores (imbalanced training split), **NOT the canonical held-out test F1**. They cannot be directly compared to the paper's canonical numbers.

#### CIC-IoT2023 — Simple Graph Pipeline (Haiku, May 15)
- **Policies**: `urg_count < 3` (F1 0.857), `flow_duration < 5` (F1 0.715), `Header_Length < 40000` (F1 0.677) — only 3 policies stored (graph hit `GraphRecursionError` after iteration 1)
- **Training-feedback macro F1**: 0.8419 (majority vote, 3 policies)
- **Assessment**: Notebook did NOT run cleanly. Only 1 round completed. Cannot be used as a paper result without fixing the recursion error and re-running all 5 rounds.

#### CIC-IoT2023 — Complex Graph Pipeline (Haiku, May 18)
- **21 expressions stored**, but all are near-identical variants of `urg_count < X and (flow_duration < Y or Number < Z)` with F1 ≈ 0.868
- Training-feedback macro F1: R1=0.8187, R2-5=0.8623 (converged, no improvement after round 2)
- **Diversity filter failed**: Despite the 5% disagreement threshold in `complex_lt_memory_write_tool`, the expressions look syntactically different but produce almost identical predictions — threshold is too loose
- **Performance vs simple pipeline**: Complex training-feedback F1 (0.8623) < Simple paper F1 (0.9426). Even accounting for evaluation-set differences, this is NOT an improvement. Complex pipeline on CIC-IoT is a failure so far.

#### CIC-IoT2023 — Simple Graph Pipeline (Gemini 2.5 Flash, May 12–13)
- **Policies**: `urg_count < 9` (F1 0.8681), `flow_duration < 8` (F1 0.797), `Duration < 70` (F1 0.747), `Header_Length < 50000` (F1 0.718)
- Training-feedback macro F1: 0.8255 (final round)
- **Assessment**: Gemini finds similar policies to Haiku but with slightly higher individual F1 (0.8681 vs 0.8573 for `urg_count`), yet the ensemble is weaker (0.8255 vs 0.8419). This is expected — the graph completed cleanly but is not better than Haiku in ensemble mode.

#### TON_IoT — Simple Graph Pipeline (Haiku, May 15)
- **Policies (9 stored)**: `proto == tcp` (F1 0.831), `service != dns` (F1 0.808), `dst_port != 53` (F1 0.768), `dst_pkts > 0` (F1 0.703), `dst_ip_bytes >= 40` (F1 0.703), weaker features below
- Training-feedback macro F1: R1=0.7741, R2=0.8081, R3=0.8058, R4=**0.8352**, R5=0.8352
- **Assessment**: Converged at R4. 5 rounds completed cleanly. Improvement pattern matches the paper's canonical result (which shows best at R4 too). These policies target protocol/port/packet-count patterns. Note the R4 training-feedback F1 (0.8352) is much higher than the paper's canonical R4 test F1 (0.7701) — training vs test split difference.

#### TON_IoT — Simple Graph Pipeline (Gemini 2.5 Flash, May 15)
- **Policies (7 stored)**: `proto == tcp` (F1 0.831), `service != dns` (F1 0.808), `dst_port != 53` (F1 0.768), `conn_state != SHR` (F1 0.539), `src_pkts >= 1` (F1 0.532)
- Training-feedback macro F1: R1=0.6461, R2-5=0.7767 (converged at R2)
- **Assessment**: Gemini converges faster but lower than Haiku. Both find the same dominant features (`proto == tcp`, `service != dns`). No improvement after R2.

#### Bot-IoT — Simple Graph Pipeline (Haiku, May 15)
- **Policies**: `N_IN_Conn_P_DstIP > 20` (F1 0.507), `dport != 53` (F1 0.500), `srate > 0.1` (F1 0.473), `seq > 30000` (F1 0.461), `stddev > 0` (F1 0.455) — 9 policies total, all weak
- Training-feedback macro F1: R1=0.4489, R5=**0.4968** (minimal improvement)
- **Root problem**: Bot-IoT has 74 normal vs 586,889 attack samples in the test split. Macro F1 on this imbalanced split is structurally near 0.5 regardless of rule quality. The paper's 0.9232 result comes from a balanced 10,000-sample evaluation that isn't being reproduced here.
- **Assessment**: The notebook evaluates on the wrong (unbalanced) test split. Results are misleading. The simple graph pipeline for Bot-IoT does not replicate the paper's experimental setup.

#### Bot-IoT — Complex Graph Pipeline (Haiku, May 18)
- **3 expressions only**: `(srate > 0.2) and (mean > 1.3) and (dport != 53)` (F1 0.395), two more at F1 0.27 and 0.27
- Training-feedback macro F1: R1=0.2846 — only 1 iteration, graph stopped
- **Failure root causes** (confirmed from earlier memory):
  1. Uses low-importance features (`srate`, `mean`, `stddev`) which have near-zero permutation importance for this dataset
  2. AND-combination is wrong for a 99.987% attack-rate dataset — needs high recall, not precision
  3. Diversity filter rejected most candidate expressions, leaving only 3
  4. Exception in iteration 2 prevented further rounds
- **Assessment**: Complete failure. Complex pipeline should NOT be run on Bot-IoT as currently designed.

#### WUSTL-IIoT — Simple Graph Pipeline (Haiku/Gemini)
- **No run found in `7-evaluation-*` notebook outputs**. The canonical results (macro F1 0.9713) come from notebook `4`. WUSTL-IIoT has the cleanest separability (`DIntPkt` dominates 95.8% permutation importance) so performance is unlikely to improve further with graph pipeline.

#### UNSW-NB15 — Simple Graph Pipeline (Haiku, May 15)
- **Policies (13 stored)**: `rate > 20000` (F1 0.719), `state == INT` (F1 0.717), `sload > 50000000` (F1 0.707), `sinpkt < 0.05` (F1 0.673), and 9 more weak policies
- Training-feedback macro F1: R1=0.7025, R2=0.7057, R3=0.7057, R4=**0.7435**, R5=0.7435
- **Assessment**: Converged at R4. Improvement from R1 (0.7025) to R4 (0.7435) is modest. The canonical paper number (0.7507) is slightly better, from the standardised test evaluation.

---

### Honest Assessment: What's Good vs Bad for IEEE Transactions

#### What's publication-quality (can be submitted now):
1. **The simple pipeline (notebook 4)** — 5-dataset, 5-round evaluation, properly documented in `results.tex`. Macro F1 results are solid and the reasoning about why each dataset behaves the way it does is coherent.
2. **Adversarial robustness (Kerckhoffs)** — Rigorous, multi-seed, reproducible.
3. **Anonymization ablation** — Clean finding (semantic names don't help, sometimes hurt).
4. **Zero-day detection** — Good variation across datasets, well-explained by feature-space structure.
5. **Inference latency** — 4.5×–80× faster than RF is a strong deployment argument.
6. **Edge deployment data** — Pi 4 and Pi 5 benchmark results exist (`pi_results/` from May 13). The `edge_deployment_evaluation.tex` section is written. This is **the key differentiator** for IEEE IoT Journal.

#### What's weak or missing:
1. **`fig_comparison_summary.pdf` is missing** from `latex/figures/`. Referenced by `results.tex` at line 232. Must be regenerated before compiling the paper.
2. **Edge section not integrated** — `edge_deployment_evaluation.tex` exists but is NOT included in `results.tex`. Pi results exist but no table or figure generated from them.
3. **Complex pipeline is failing** — CIC-IoT: diversity collapse (21 near-identical expressions). Bot-IoT: structural failure. WUSTL/TON/UNSW: not run. The complex pipeline should either be fixed and run on all 5 datasets with proper canonical evaluation, OR dropped from the paper entirely.
4. **Graph pipeline (notebook 7) not canonically evaluated** — The `7-evaluation` runs use different test splits (imbalanced training split) vs the canonical balanced 10,000-sample test that feeds the paper. Until the policies from notebook 7 are evaluated on the canonical test, we don't know if the improved graph is actually better than notebook 4.
5. **Gemini comparison not systematically documented** — Gemini runs exist for CIC-IoT, TON-IoT, Bot-IoT. But no side-by-side table or figure comparing Haiku vs Gemini is in the paper.
6. **TON_IoT attack recall is critically bad** — 0.243 (24.3%). This is the worst result in the paper and will be the first thing reviewers flag. For a real IDS, missing 75% of attacks is unacceptable. The paper explains this as "intra-class overlap" but reviewers will push for at least a partial fix.
7. **`results_evaluation_new.tex`** — A partially-complete older draft that's been superseded. Should be archived or deleted to reduce confusion.

---

### Edge Deployment Status (High Priority for IoT Journal)

- Pi 4 and Pi 5 benchmarks run on May 13, 2026. CSVs and JSONs are in `pi_results/`.
- `edge_deployment_evaluation.tex` describes the methodology correctly.
- **Missing**: Table and figure generated from the Pi results. No code to parse the Pi benchmark JSON and produce a latex table has been written yet.
- The Pi benchmark evaluates LLM policy, DT, and RF in streaming (1 row) and batch mode.
- This section needs: a latency table (median μs/row by device × mode × classifier) and a figure.

---

### Next Steps (Priority Order)

**P0 — Unblock paper compilation:**
- Regenerate `fig_comparison_summary.pdf` (binary F1 + latency side by side for all 5 datasets). Data is already in the paper's tables.

**P1 — Integrate edge deployment:**
- Parse `pi_results/Raspberry Pi 4/*.json` and `Raspberry Pi 5/*.json`
- Generate a latency table (Pi 4 vs Pi 5, streaming vs batch, LLM vs DT vs RF)
- Generate a figure
- Insert `edge_deployment_evaluation.tex` into `results.tex` and add table + figure

**P2 — Canonically evaluate notebook 7 policies:**
- Export policies from each dataset's `7-evaluation` SQLite DB to JSON rule files
- Run the standardised balanced 10,000-sample test evaluation on them (same protocol as the `attack-precision-report` files)
- Compare against notebook 4's canonical results — determine if graph pipeline is actually better

**P3 — Fix or drop the complex pipeline:**
- Option A (Fix): Address diversity collapse by using Hamming distance instead of prediction disagreement; reduce AND-combination bias for high-skew datasets; fix the iteration-2 exception
- Option B (Drop): Remove complex pipeline from paper scope entirely. The current evidence (CIC-IoT complex F1 ≈ simple F1, Bot-IoT failed, others not run) does not support including it in a paper without more work.

**P4 — Address TON_IoT recall problem:**
- The 0.243 attack recall is a liability for reviewers. Consider: (1) a lower threshold variant that trades precision for recall, (2) an ensemble with a high-recall rule added, (3) reframing as a precision-first IDS where false negatives are expected for this specific dataset type.

**P5 — Target venue decision:**
- IEEE IoT SI-SMCC deadline missed (May 15). Next options:
  - IEEE Transactions on Dependable and Secure Computing (TDSC) — quarterly, no fixed deadline
  - IEEE Internet of Things Journal — regular track, fits the deployment angle strongly
  - ACM Computing Surveys — if scope expanded to survey-style treatment
- Decision needed before investing in further experiments.

---

## 📅 [2026-05-15] — CIC-IoT LLM Graph Run: Policy Memory + Evaluation Notes

**Source(s):** `1-cic-iot/7-evaluation-2-class-llm-graph-improved-again.ipynb`, `1-cic-iot/results/database.db`, `1-cic-iot/results/policies-2026-05-15-16-48-52-claude-haiku-4-5-20251001.txt`

### Run Notes
- Fixed feature-name mismatch issue: model generated `Tot_size`, dataset column is `Tot size`.
- Added feature-name resolver so spacing/underscore aliases do not crash policy evaluation.
- Run produced valid saved policies and a feedback evaluation.
- Run hit `GraphRecursionError`; graph did not terminate cleanly — only 1 round completed.

### Saved Policies (Training-feedback F1, imbalanced split)

| Feature | Operator | Value | Train macro F1 |
|---------|----------|-------|----------------|
| `urg_count` | `<` | `3` | `0.8573` |
| `flow_duration` | `<` | `5` | `0.7148` |
| `Header_Length` | `<` | `40000` | `0.6765` |

### Test Evaluation (on imbalanced training split)
- Majority-vote macro F1: `0.8419`

### Concerns
- Only 3 policies (not k=5). GraphRecursionError cut the run short.
- Evaluation is on the raw imbalanced test split (97.65% attack), NOT the canonical balanced test. Numbers are NOT comparable to the paper's 0.9426 canonical result.

---

## 📅 [2026-04-28 11:32] — Kerckhoffs Robustness: ESR vs. Perturbation Budget (All Conditions)

**Source(s):** `iot-llm-hids-pg/experiments/10-kerckhoffs-robustness.ipynb`, `experiments/results/kerckhoffs/kerckhoffs-results-2026-04-28-11-32-41.json`

### Summary
This experiment measures the adversarial evasion robustness of five IDS detection rule conditions — LLM_orig, LLM_anon, DT_d3, DT_d5, RF_d5 — across two IoT network datasets (CIC-IoT2023 withheld class: Mirai-udpplain; WUSTL-IIoT withheld class: Reconn). The core metric is the Evasion Success Rate (ESR) as a function of perturbation budget ε (in training std units), with the primary summary statistic being ε at ESR=0.5 (median evasion cost). All conditions are evaded at very small perturbation budgets (all ε@ESR=0.5 < 0.05), confirming that threshold-based IDS rules — whether LLM-generated or tree-learned — share a structural vulnerability to minimal feature perturbations. The most robust condition across both datasets is DT_d3 on WUSTL-IIoT (ε@ESR=0.5 = 0.03).

### Key Findings

**Median Evasion Cost (ε at ESR=0.5, std units):**

| Dataset     | LLM_orig | LLM_anon | DT_d3 | DT_d5 | RF_d5 |
|-------------|----------|----------|-------|-------|-------|
| CIC-IoT2023 | 0.008    | 0.002    | 0.005 | 0.005 | 0.008 |
| WUSTL-IIoT  | 0.001    | 0.020    | 0.030 | 0.001 | 0.001 |

*(Exact values from JSON: CIC-IoT LLM_orig = 0.007851, LLM_anon = 0.002113; WUSTL-IIoT LLM_anon = ~0.02, DT_d3 = ~0.03)*

**Detection rates (% of attack pool flagged):**
- CIC-IoT2023 attack pool: 7,879 instances
  - LLM_orig: 5,876 flagged (74.6%) | LLM_anon: 7,251 (92.0%) | DT_d3: 7,699 (97.7%) | DT_d5: 7,740 (98.2%) | RF_d5: 7,686 (97.5%)
- WUSTL-IIoT attack pool: 15,756 instances
  - LLM_orig: 11,548 flagged (73.3%) | LLM_anon: 11,929 (75.7%) | DT_d3: 15,747 (99.9%) | DT_d5: 15,755 (100.0%) | RF_d5: 15,748 (100.0%)

**Top targeted features at ε=1.0:**
- CIC-IoT / LLM_orig: `Tot sum` (5,524×), `Header_Length` (4,982×), `Rate` (3,002×)
- CIC-IoT / LLM_anon: `Covariance` (6,523×), `Header_Length` (5,006×), `Rate` (4,003×)
- CIC-IoT / DT_d3–d5: `rst_count` dominates (7,648–7,668×)
- WUSTL-IIoT / LLM_orig: `TotBytes` (11,547×), `SrcPkts` (7,203×), `DstPkts` (7,187×)
- WUSTL-IIoT / LLM_anon: `SrcPkts` (11,929×), `DstPkts` (10,236×), `Dport` (4,137×)
- WUSTL-IIoT / DT_d3: `DIntPkt` (15,726×) is overwhelmingly dominant

**Multi-seed validation (seeds 42, 123, 456):** All ε@ESR=0.5 values have std ≈ 0.00, confirming full reproducibility across train/test splits.

### Anomalies / Warnings
- ⚠️ **DT_d5 on WUSTL-IIoT is paradoxically fragile**: ε@ESR=0.5 = 0.00 (ESR=1.000 at ε=0.001), while the shallower DT_d3 achieves ε=0.03. The deeper tree creates more leaf paths with sharper, tighter thresholds that are individually cheap to violate.
- ⚠️ **RF_d5 on WUSTL-IIoT also collapses early**: ESR=0.996 at ε=0.001, suggesting WUSTL-IIoT features have low std variance.
- ⚠️ **LLM_anon detection rate is higher than LLM_orig on both datasets**, yet LLM_anon is less robust on CIC-IoT (ε=0.002 vs 0.008). Higher recall ≠ higher robustness.

### Notes for Paper
- `results/kerckhoffs/robustness_curves.png` is ready for inclusion (log-scaled x-axis already done).
- `results/kerckhoffs/feature_targeting.png` suitable for appendix — shows LLM rules distribute perturbation cost across multiple features while tree rules concentrate on a single bottleneck.
