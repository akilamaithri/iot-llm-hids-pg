# Implementation Plan: results_evaluation_new.tex

## Group 2 — Missing UNSW-NB15 Experiments (Must Run)
The 5-unsw-nb15/results/ directory has only ML baseline .txt files and a generated rules .txt. No LLM classification JSON, no anonymization ablation JSON, no zero-day JSON exist. Three cells in the paper depend on this:

Line 126: [INSERT: anon_f1_unsw]
Line 159: [INSERT: zdr_unsw] (both Known-F1 and ZDR)
Line 206: [INSERT: f1_orig_unsw] and [INSERT: f1_anon_unsw]
2.1 — Run binary LLM experiment for UNSW-NB15

Open 5-unsw-nb15/4-evaluation-2-class-llm.ipynb and run it end-to-end. This produces the LLM F1 (original). Target output: a JSON result file analogous to the other datasets' ablation JSONs.

2.2 — Run anonymization ablation for UNSW-NB15

Run 5-unsw-nb15/4-evaluation-2-class-llm-anon.ipynb (if it exists) or adapt the CIC-IoT/WUSTL anon notebook pattern to produce 5-unsw-nb15/results/anon/unsw-nb15-anon-ablation.json.

2.3 — Run zero-day experiment for UNSW-NB15

Run 5-unsw-nb15/8-zero-day-evaluation.ipynb. This produces Known-F1 and ZDR. You'll also need to decide which class to withhold (the dataset is binary attack/normal, so zero-day is less applicable — you may need to document why UNSW zero-day results are --- rather than leaving [INSERT] hanging).

Recommendation: If UNSW-NB15 is inherently binary (no multi-class), the zero-day experiment is not meaningful (you'd be withholding "attack" or "normal"). Consider replacing [INSERT: zdr_unsw] with --- and a footnote explaining that UNSW-NB15 has no named sub-classes, making zero-day evaluation undefined. This is defensible and honest.

## Group 3 — Missing Section: Kerckhoffs / Adversarial Robustness
There is no section in the tex for the Kerckhoffs results, yet there are:

4 result JSONs in experiments/results/kerckhoffs/
2 PNGs: robustness_curves.png and feature_targeting.png
The latest result (2026-04-28-14-52-13.json) covers CIC-IoT and WUSTL, with conditions LLM_orig, LLM_anon, DT_d3, DT_d5, RF_d5 across epsilon values [0, 0.001, …, 5.0]
3.1 — Add \subsection{Adversarial Robustness (Kerckhoffs Evaluation)} after Section 4 (Threshold Sensitivity). Include:

A brief explanation of the Kerckhoffs threat model (attacker knows the rules, can perturb features)
\includegraphics for robustness_curves.png and feature_targeting.png
A table with epsilon_at_05 values per dataset/condition (already in the JSON)
Key finding: at what ε does each method fail, and how does LLM compare to DT/RF
Group 4 — Reviewer Red Flags (Quality Fixes)
4.1 — LLM multiclass F1 = 0.45 vs RF F1 = 0.89 (2× gap)

A reviewer will immediately flag this. The paper has no narrative in the tex explaining the gap beyond the table itself. You need to add a paragraph after tab:multiclass_rf_cic that:

Frames the LLM as a zero-shot reasoner (no gradient training), making 0.45 vs 0.89 a different comparison than two trained classifiers
Highlights where the LLM is close (DDoS-Fragmentation: 0.83 vs 0.99, Mirai: 0.85 vs 0.99) vs. where it collapses (DNS_Spoofing: 0.14, BenignTraffic: 0.31 — the LLM cannot distinguish "normal" traffic structurally)
Notes that DNS_Spoofing has only 81 test samples — the gap there is partly a support problem
4.2 — Bot-IoT ablation test set is only 148 samples (74 per class)

The ML baseline uses 10,000-sample corpus (2,000 test). The ablation JSON shows test support of 148. This is a serious methodological inconsistency a reviewer will catch. Action: check 4-bot-iot/4-evaluation-2-class-llm-anon.ipynb to understand why the test set is so small, then either:

Re-run with a properly sized test set matching the other datasets, or
Add a footnote in the table acknowledging the small test set and its implication for the std values
4.3 — CIC-IoT ZDR = 0.0004 for Mirai-udpplain (near-zero)

The existing narrative covers this (feature overlap argument), but the tex should add a sentence noting this is an expected failure mode by design — the Mirai-udpplain class is statistically near-indistinguishable from the benign class at the feature level sampled. This converts a "bad result" into a "diagnostic finding."

4.4 — WUSTL-IIoT has 4 classes but only binary results

The footnote says multi-class experiments were not conducted, but gives no reason. Add a brief parenthetical (e.g., "multi-class WUSTL excluded due to severe class imbalance in the subsampled corpus") to pre-empt reviewer questions.

Group 5 — Unused Figures (Already Generated, Not Referenced)
These figures exist in charts/ but are never \includegraphics'd in the tex:

anon_ablation_summary.png — 4-panel overview of all ablation results
anon_ablation_consistency.png — per-seed feature selection consistency
anon_ablation_tradeoff.png — Jaccard vs ΔF1 tradeoff matrix
anon_ablation_features.png — feature divergence heatmap
zero_day_eval_summary.png — summary table for zero-day results
5.1 — In Section 4 (Anonymization Ablation), add a figure using anon_ablation_summary.png to give a visual overview. The text already describes these results thoroughly; the figure strengthens it.

5.2 — In Section 3 (Zero-Day), consider adding zero_day_eval_summary.png as a compact visual summary alongside the table.

Group 6 — Appendix Item Referenced but Absent
Line 175 references Table~\ref{tab:zeroday_wustl_both} in Appendix~\ref{app:detailed}. Check appendix.tex — if this table is missing, add it showing both WUSTL zero-day runs (ZDR=1.0 and ZDR=0.999) from the two JSON files: zeroday-Reconn-seed42-2026-04-20-17-07-34.json and zeroday-Reconn-seed42-2026-04-20-17-58-16.json.

Priority Order (start here)
#	Task	Blocker?	Effort
2.3	Decide UNSW zero-day policy (run or document as N/A)	Yes — paper has [INSERT]	Low
2.1/2.2	Run UNSW-NB15 LLM + anon ablation	Yes — paper has [INSERT]	High
1.1/1.2	Fill sample counts from notebooks	Yes — paper has [INSERT]	Low
4.1	Add narrative for LLM vs RF gap	Reviewer will flag	Medium
4.2	Fix or footnote Bot-IoT 148-sample issue	Reviewer will flag	Medium
6	Write tab:zeroday_wustl_both in appendix	Referenced but absent	Low
3.1	Add Kerckhoffs section	Results exist, section missing	High
5.1/5.2	Insert existing chart figures	Results exist, unused	Low
4.3/4.4	Add explanatory text for edge case results	Polish	Low
