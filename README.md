# LLM-driven Policy Generation for IoT Intrusion Detection Systems

## Introduction

In this research we try to find how Large Language Models (LLMs) perform to improve the IoT security and access control. In particular, we investigate whether LLMs can generate effective policies (rules) to detect intrusion attacks in IoT systems. This experiment integrates Retrieval-Augmented Generation (RAG) and function calling capabilities of LLMs to build a novel policy generation framework for IoT Intrusion Detection Systems.

## Getting Started

1. Clone the repository.
2. Install required Python libraries using `pip install -r requirements.txt`.
3. Create `.env` file in the root directory and add API keys as follows.

    ```bash
    OPENAI_API_KEY=
    GOOGLE_API_KEY=
    ANTHROPIC_API_KEY=
    LANGCHAIN_API_KEY=
    LANGCHAIN_PROJECT=
    LANGCHAIN_TRACING_V2=
    ```

4. Create `data` directory in the root directory and sub directories for datasets as follows.

    ```
    data
    |-cic-iot
    |-wustl-iiot
    |-ton-iot
    |-bot-iot
    └-unsw-nb15
    ```

5. Place downloaded datasets in the relevant sub directories.

6. Run Python notebooks in order for each dataset.

    > For ex:
    >
    > Run `00-dataset-analysis.ipynb`, `01-preprocessing.ipynb`, `02-baseline-ml.ipynb`, `03-multiclass-classification-llm.ipynb`, `04-zero-day-detection-claude.ipynb`, `05-anonymization-ablation-original.ipynb`, `06-binary-classification-llm-claude.ipynb`, ... in `1-cic-iot` directory to evaluate `CICIoT2023` dataset. Each dataset folder follows the same numbering, matching the order results appear in Section V of the paper (`03-multiclass-classification-llm.ipynb` only exists for CIC-IoT2023, WUSTL-IIoT, and UNSW-NB15; `-gemini` siblings provide the cross-vendor comparison).

## Repository Layout

```
RAG Paper/
├─ iot-llm-hids-pg/                 # this repository — notebooks + pipeline code only
│  ├─ 1-cic-iot/ … 5-unsw-nb15/     # one notebook per paper result, per dataset (see above)
│  ├─ experiments/                  # cross-dataset scripts (see below)
│  └─ lib/                          # shared policy-pipeline code used by experiments/
├─ iot-llm-hids-pg-results/         # generated per-dataset results/ folders (gitignored;
│  └─ 1-cic-iot/ … 5-unsw-nb15/     #  rule outputs, feature-importance reports, plots — not code)
└─ iot-llm-hids-pg-archive/         # superseded/dev-history notebooks and scripts, kept for
   ├─ 1-cic-iot/ … 5-unsw-nb15/     #  reproducibility but not referenced by the published paper
   └─ experiments/                  #  (early pipeline iterations, hyperparameter-sweep tooling,
                                     #   the abandoned "hybrid RAG" exploration, etc.)
```

`experiments/` only keeps what feeds a table or figure in the paper:

| File | Paper reference |
| - | - |
| `07-adversarial-robustness-kerckhoffs.ipynb` (+ `_make_notebook.py` generator) | §V-G, Table IX, Fig. 6 |
| `08-edge-benchmark-matched-inference.py`, `08-edge-benchmark-raspberry-pi.py` | §V-H, Table X |
| `run_pipeline.py`, `multi_seed.py`, `eval_policy.py` | Underlying driver/eval for the cross-vendor binary results |
| `build_results_table.py`, `aggregate_multi_seed.py`, `standardize-results.py` | §V-E/B/D, Tables III, VI, VII, VIII |
| `policies/`, `results/` | Generated artifacts consumed/produced by the scripts above |

Hyperparameter-sweep tooling (`sweep_policy.py`, `build_sweep_summary.py`), the standalone `check_imbalances.py` diagnostic, and the abandoned hybrid-RAG comparison files live in `iot-llm-hids-pg-archive/experiments/` — the paper uses globally fixed hyperparameters (k=5, n_top=5, n_rounds=8), so no sweep ablation is reported.

## Datasets

| Name | Paper(s) | Year |
| - | - | - |
| CICIoT2023* | CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment | 2023 |
| Edge-IIoTSet | Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning | 2022 |
| WUSTL-IIoT* | WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research | 2021 |
| IoT-23 | IoT-23: A labeled dataset with malicious and benign IoT network traffic | 2020 |
| TON_IoT* | TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems | 2020 |
| Bot-IoT* | Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset | 2019 |
| N-BaIoT | N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders | 2018 |
| UNSW-NB15* | UNSW-NB15: A Comprehensive Data set for Network Intrusion Detection systems | 2015 |

## Large Language Models

| Name                      | Provider  |
|---------------------------|-----------|
| claude-haiku-4-5*         | Anthropic |
| gemini-2.5-flash*         | Google    |

\* Used as the two cross-vendor rule generators reported in the paper (Tables VI/VII). Earlier exploratory runs against `gpt-4o`, `gemini-1.5-pro`, and `claude-3-5-sonnet` are preserved in `iot-llm-hids-pg-archive/`.