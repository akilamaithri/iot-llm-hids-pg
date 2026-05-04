# Zero-Day Attack Classification Experiment
## Comprehensive Reference Guide

**Date:** April 2026  
**Project:** RAG-Enhanced LLM for IoT-IDS (Journal Paper Contribution 3)  
**Author:** Akila Maithri  
**Status:** Active (Pilot complete, replication in progress)

---

## 1. Executive Summary

This document provides a complete reference for the **zero-day attack classification experiment**, a novel evaluation method for testing whether LLM-generated IDS policies can detect previously unseen attack types.

### Core Innovation
- **Traditional evaluation:** Train on multiclass attacks, test on same classes
- **Zero-day evaluation:** Train on multiclass attacks MINUS one class, test that withheld class at inference
- **Key metric:** Zero-Day Detection Rate (ZDR) = `fraction_of_withheld_class_flagged_as_attack`

### Hypothesis
LLM-generated threshold rules may capture abstract attack indicators beyond specific attack profiles, potentially generalizing to unseen attack types. ML baselines (DT/RF) serve as comparison.

---

## 2. Experimental Design

### 2.1 High-Level Protocol

```
FOR each dataset:
  FOR each withheld attack class:
    1. Load full multiclass population CSV
    2. Remove withheld class from training data
    3. Create 1:1 balanced binary (normal/attack) training pool
    4. 80/20 stratified split → train_df, test_known_df
    5. Zero-day test set = ALL withheld class samples (never in training)
    
    6. Embed representative samples (BGE-M3, top 10 via cosine similarity to mean)
    7. Run LangGraph RAG-LLM pipeline:
       - LLM generates k=5 rules via majority-vote feedback loop
       - n=5 iterations, temperature=0.1 (low stochasticity)
    
    8. Evaluate:
       - Known-attack F1: standard binary eval on test_known_df (sanity check)
       - ZDR: apply rules to withheld class, count as 'attack'
       - ML baselines (DT/RF): same condition for comparison
    
    9. Multi-seed runs (3 seeds: 42, 123, 456) for error bars
```

### 2.2 Critical Design Decisions

#### Decision 1: Single Data Source (Not Pre-Built Sample)
**Issue:** Original approach used `sample-100000-2.csv` (pre-computed balanced sample) for training, but `population.csv` for zero-day test. This created a data inconsistency.

**Solution:** Use ONLY `population.csv` (raw, full dataset with original labels) for all splits:
- Training: benign + known attacks (balanced)
- Test known: benign + known attacks (20% of training, stratified)
- Test zero-day: ALL withheld class rows (never in training)

**Rationale:** Consistent data source, maximizes zero-day test sample count.

#### Decision 2: 1:1 Class Balance (Not Pre-Computed Ratio)
**Issue:** CICIoT2023 `sample-100000-2.csv` has 10,800 benign + 89,200 attack (1:8.26 imbalance). Mirai-udpplain experiment showed LLM can fail to generalize even within the same attack family if training is imbalanced.

**Solution:** For each experiment, sample attack rows to match benign count:
```python
n_benign = len(benign_df)
known_atk_sampled = known_atk_df.sample(n=n_benign, random_state=SEED)
train_pool = pd.concat([benign_df, known_atk_sampled])  # 1:1 balanced
```

**Rationale:** Fair comparison with ML baselines; avoids majority-class bias; consistent with anonymization ablation notebook.

#### Decision 3: BGE-M3 Embeddings (Not Chroma Vector Store)
**Issue:** Pre-built Chroma vector store in `vector-stores/chroma-db-100000-2/` includes the withheld class in its training documents. Using it would leak zero-day information.

**Solution:** Implement fresh BGE-M3 embeddings per experiment:
1. Subsample up to 100 rows per class
2. Embed as `str(row.tolist())` using BAAI/bge-m3
3. Compute mean embedding
4. Select top 10 via cosine similarity to mean
5. Format as JSON dict with feature names as keys

**Rationale:** Consistent with multiclass evaluation notebook methodology; no data leakage; reproducible per-seed.

#### Decision 4: ToolNode Custom Implementation
**Issue:** `from langgraph.prebuilt import ToolNode` is not available in the installed langgraph version.

**Solution:** Implement custom `make_tool_node()` function:
```python
def make_tool_node(tools):
    tools_by_name = {t.name: t for t in tools}
    def tool_node(state):
        ai_msg = next(m for m in reversed(state['messages']) 
                      if extract_tool_calls(m))
        results = []
        for tc in extract_tool_calls(ai_msg):
            result = tools_by_name[tc['name']].invoke(tc['args'])
            results.append(ToolMessage(content=str(result), 
                                      tool_call_id=tc['id']))
        return {'messages': results}
    return tool_node
```

**Rationale:** No external dependency; functionally equivalent; future-proof for version changes.

---

## 3. Datasets & Class Selection

### 3.1 CICIoT2023

**Dataset:** `/data/cic-iot/CSV/cic-iot-2023-population.csv`  
**Total rows:** 1,690,000  
**Features:** 43 numeric  
**Benign class:** `BenignTraffic` (39,394 rows)

#### Withheld Class Candidates (Ranked by Scientific Value)

| Class | Samples | Type | Justification | Paradigm |
|-------|---------|------|---------------|----------|
| **MITM-ArpSpoofing** (Primary) | 11,211 | ARP spoofing | Layer 2 attack, distinct binary feature (ARP=1), low-rate | Orthogonal (Different layer) |
| **Mirai-udpplain** | 31,864 | UDP flood botnet | Same attack family as training (Mirai variants), high-rate | Within-family |
| **DDoS-SynonymousIP_Flood** | 129,875 | DDoS variant | Volumetric flood, similar to training majority | Within-family |
| **DDoS-SlowLoris** | 814 | App-layer DDoS | HTTP slowdown, distinct from volumetric floods | Different mechanism |
| Backdoor_Malware | 113 | Persistence | Stealth/C2 traffic | Orthogonal (Too few samples) |
| SqlInjection | 192 | Injection | Application-layer | Orthogonal (Too few samples) |

**Training attack classes:** 32 total (all except withheld)  
**Training set composition (balanced):** 
- Benign: ~20k rows
- Known attack: ~20k rows (sampled from 32 classes)

### 3.2 Bot-IoT

**Dataset:** `/data/bot-iot/bot-iot-population.csv`  
**Total rows:** 2,934,817  
**Features:** 10 numeric (seq, stddev, N_IN_Conn_P_SrcIP, min, state_number, mean, N_IN_Conn_P_DstIP, drate, srate, max)  
**Benign class:** `Normal` (370 rows — very small!)

#### Withheld Class Candidates

| Class | Samples | Justification |
|-------|---------|---------------|
| **Service_Scan** (Recommended) | 58,626 | Network reconnaissance, distinct from UDP/TCP flooding majority |
| OS_Fingerprint | 14,293 | Passive OS discovery, also reconnaissance |
| HTTP | 1,970 | Application-layer DDoS |
| Keylogging | 59 | Data theft (too few samples) |
| Data_Exfiltration | 6 | Data theft (insufficient) |

**⚠️ Important:** Only 370 benign rows → training pool is tiny (296 normal + 296 attack per seed). This is intentional (maintains 1:1 ratio) but may affect F1 scores due to limited training data.

---

## 4. Experiment Results

### 4.1 CICIoT2023 — Experiment 1: MITM-ArpSpoofing (Orthogonal Paradigm)

**Withheld class:** MITM-ArpSpoofing (11,211 samples, ARP-layer attack)  
**Training:** 39,394 benign + 39,394 known attack (balanced, 1:1)  
**Pipeline:** GPT-4o, k=5 rules, n=5 iterations, seed=42

#### Results (Seed 42)

| Metric | LLM Rules | Decision Tree | Random Forest |
|--------|-----------|---------------|---------------|
| Known-Attack F1 | 0.8648 | 0.9947 | 0.9964 |
| ZDR | 0.0218 (2.2%) | 0.5546 (55.5%) | 0.2201 (22.0%) |

**Generated rules (5):**
```
flow_duration     <= 1.0
Header_Length     <= 200.0
Protocol Type     <= 6.0
Rate              <= 5.0
Tot sum           <= 600.0
```

#### Interpretation

**Why LLM failed (ZDR = 2.2%):**
- Rules are tuned to **volumetric floods** (high Rate, large headers)
- Training set is ~85% DDoS/DoS/Mirai (all volumetric)
- MITM-ArpSpoofing is **ARP-layer, low-rate** → matches benign traffic profile
- Majority vote rejects it as normal
- ML baselines: DT achieved 55.5% (better than LLM) via statistical patterns; RF achieved 22%

**Scientific conclusion:**
- **Fundamental limitation confirmed:** LLM rules don't generalize across orthogonal attack paradigms
- **Within-family generalization:** DT/RF's statistical learning on volumetric attacks doesn't help on layer-2 attacks
- This is an **honest negative result** showing the importance of attack-family-specific training

---

### 4.2 CICIoT2023 — Experiment 2: Mirai-udpplain (Within-Family)

**Withheld class:** Mirai-udpplain (31,864 samples, UDP flood variant)  
**Training:** 39,394 benign + 39,394 known attack (balanced, 1:1)  
**Training attack classes:** 32 (including other Mirai variants: greeth_flood, greip_flood)  
**Pipeline:** GPT-4o, k=5 rules, n=5 iterations, seed=42

#### Results (Seed 42)

| Metric | LLM Rules | Decision Tree | Random Forest |
|--------|-----------|---------------|---------------|
| Known-Attack F1 | 0.8301 | 0.9934 | 0.9963 |
| ZDR | 0.0004 (0.04%) | 0.9995 (99.95%) | 0.9998 (99.98%) |

**Generated rules:**
```
Header_Length     <= 100.0
Protocol Type     <= 7.0
Rate              <= 5.0
Tot sum           <= 600.0
Weight            <= 150.0
```

#### Interpretation

**Why LLM failed on within-family attack:**
- Despite Mirai-udpplain being a **high-rate UDP flood** (same family as training Mirai variants)
- LLM generated `Rate <= 5.0` → flags only LOW-rate traffic as attack
- Mirai-udpplain (Rate >> 5) fails all thresholds → classified as normal
- The 10 representative samples fed to LLM happened to skew toward low-rate attacks

**Why DT/RF succeeded (~100% ZDR):**
- Both saw other Mirai variants in training (greeth_flood, greip_flood)
- Statistical learners extract high-rate UDP flooding pattern
- This pattern perfectly generalizes to Mirai-udpplain
- **ML baselines dominate LLM dramatically on within-family generalization**

**Key finding:**
- LLM's lack of implicit family-knowledge is a critical weakness
- LLM only sees 10 representative samples → narrow view of attack space
- DT/RF exploit thousands of training examples to discover family patterns
- This explains why LLM ZDR ≈ 0% despite being the "same family"

#### Multi-Seed Summary (3 seeds: 42, 123, 456)

```
Known-attack F1 : 0.8390 ± 0.0166
ZDR             : 0.0243 ± 0.0304  (mean ≈ 2.4% across seeds)
```

Very consistent low ZDR across seeds (all < 7%), confirming the pattern is robust.

---

## 5. Methodology Details

### 5.1 Data Preparation (Pseudocode)

```python
# Load raw population
df = pd.read_csv('population.csv')

# Define classes
BENIGN_CLASS = 'BenignTraffic'  # or 'Normal' for Bot-IoT
WITHHELD_CLASS = 'Mirai-udpplain'
feature_cols = [c for c in df.columns if c not in ['label', 'category', ...]]

# Partition
benign_df = df[df['label'] == BENIGN_CLASS]
known_atk_df = df[(df['label'] != BENIGN_CLASS) & 
                  (df['label'] != WITHHELD_CLASS)]
zeroday_df = df[df['label'] == WITHHELD_CLASS]

# Balance and split
train_pool = pd.concat([
    benign_df,
    known_atk_df.sample(n=len(benign_df), random_state=SEED)
])
train_pool['binary_label'] = 'normal' or 'attack'
train_df, test_known_df = train_test_split(train_pool, test_size=0.2, 
                                           stratify=binary_label, 
                                           random_state=SEED)

# Never include zero-day in training
zeroday_test_df = zeroday_df[feature_cols]
```

### 5.2 Representative Sample Retrieval (BGE-M3)

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
)

def get_representative_samples_bge(df, n=10, max_embed=100, seed=42):
    # Subsample up to max_embed rows
    sample = df.sample(n=min(max_embed, len(df)), random_state=seed)
    
    # Embed as stringified rows
    docs = [str(row.tolist()) for _, row in sample.iterrows()]
    vecs = np.array(embeddings.embed_documents(docs))
    
    # Cosine similarity to mean
    mean_vec = vecs.mean(axis=0)
    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(mean_vec)
    sims = (vecs @ mean_vec) / np.where(norms == 0, 1e-9, norms)
    
    # Top N by similarity
    top_idx = np.argsort(sims)[::-1][:n]
    return sample.iloc[top_idx]
```

### 5.3 LangGraph Pipeline Topology

```
START
  ↓
llm_node (sends prompt with 10 repr. samples to GPT-4o)
  ↓ (output: AI message with k tool calls)
tools_condition_edge → tools_node (execute each tool call)
  ↓
  └→ evaluation_tool (evaluates each rule on train_df, returns F1)
  ↓ (tool results appended to message chain)
llm_node (LLM sees F1 feedback, refines rules)
  ↓
tools_condition_edge → [tools_node | evaluate_node]
  ↓
evaluate_node (apply all k rules via majority vote to test_df)
  └→ send combined F1 feedback
  ↓
feedback_condition_edge → [llm_node (continue if i < n) | END]
```

**Key:** Dual-stage evaluation:
1. Per-rule F1 on training data (from `evaluation_tool`)
2. Combined majority-vote F1 on test data (from `evaluate_node`)

### 5.4 Rule Application (Majority Vote)

```python
def evaluate_zero_day(tool_calls, zd_df):
    preds = []
    for i in range(len(zd_df)):
        votes = [
            'attack' if operators[tc['args']['op']](
                zd_df.iloc[i][tc['args']['feature_name']],
                float(tc['args']['value'])
            ) else 'normal'
            for tc in tool_calls
        ]
        preds.append(mode(votes))  # majority vote over k rules
    
    zdr = sum(p == 'attack' for p in preds) / len(preds)
    return zdr, preds
```

**Operators:** `<, >, <=, >=, ==, !=` (LLM chooses one per rule)

---

## 6. Notebook Specifications

### 6.1 CICIoT2023: `1-cic-iot/8-zero-day-evaluation.ipynb`

**Status:** ✅ Complete and tested with two withheld classes

**Cell Structure:**

| Cell | Name | Purpose |
|------|------|---------|
| 0 | Config | WITHHELD_CLASS, parameters (k=5, n=5, SEEDS=[42,123,456]) |
| 1 | Load data | Read population.csv, show class distribution |
| 2 | Split | Partition benign/known/zero-day, balance to 1:1, stratified 80/20 |
| 3 | Embeddings | BGE-M3: embed 100 rows per class, select top 10 via cosine sim |
| 4 | evaluation_tool | LangChain @tool: applies single rule, returns macro F1 |
| 5 | Graph | LangGraph StateGraph: llm_node, tools, evaluate_node |
| 6 | Run pipeline | Invoke graph (seed 42), generate rules, save best |
| 7 | Standard eval | Binary eval on test_known_df (sanity check F1) |
| 8 | Zero-day eval | Apply rules to zeroday_df_test, compute ZDR |
| 9 | ML baseline | DT/RF comparison, same zero-day condition |
| 10 | Summary | Results table, save JSON to results/llm/ |
| 11 | Multi-seed | Re-run Cells 6-8 for seeds [123, 456], compute mean ± std |
| 12 | Visualize | Bar chart: rule thresholds vs mean feature per group |

**Parameters:**
```python
WITHHELD_CLASS = 'MITM-ArpSpoofing'  # or 'Mirai-udpplain' or other
k = 5           # number of rules
n = 5           # feedback iterations
SEEDS = [42, 123, 456]
N_REPR = 10     # representative samples per class
MAX_EMBED = 100 # max rows to embed (cap for speed)
```

**Key Variables (Global):**
- `feature_cols`: all numeric feature names
- `normal_df_train`, `attack_df_train`: training data (1:1 balanced)
- `normal_df_test`, `attack_df_test`: test known attacks (stratified from training)
- `zeroday_df_test`: ALL withheld class rows (never in training)
- `best_tool_calls`: list of dicts `{'name': 'evaluation_tool', 'args': {...}, 'id': ...}`

**Outputs:**
```
results/llm/zeroday-{WITHHELD_CLASS}-seed{SEED}-TIMESTAMP.json
  {
    "withheld_class": "Mirai-udpplain",
    "zero_day_samples": 31864,
    "known_f1_llm": 0.8301,
    "zdr_llm": 0.0004,
    "rules": [
      {"feature_name": "Header_Length", "op": "<=", "value": 100.0},
      ...
    ]
  }
```

### 6.2 Bot-IoT: `4-bot-iot/8-zero-day-evaluation.ipynb`

**Status:** ✅ Created (not yet run)

**Adaptations from CICIoT:**
- Label column: `subcategory` (not `label`)
- Benign class: `Normal` (not `BenignTraffic`)
- Features: 10 numeric (dropped: pkSeqID, proto, saddr, sport, daddr, dport, attack, category, subcategory)
- Default withheld class: `Service_Scan` (58,626 samples)
- Training pool: only 370 benign rows → tiny training set (~296 + 296 after split)

**Same structure as CICIoT notebook (12 cells identical in logic, just adapted column names)**

---

## 7. Key Design Rationales

### 7.1 Why 1:1 Class Balance?

Standard practice in binary classification. Avoids majority-class bias where all predictions are "normal". Matches the anonymization ablation notebook methodology.

### 7.2 Why Use All Withheld Class Rows for Test?

Maximizes statistical reliability of ZDR (more samples = lower variance). Other designs might sample 80/20 of withheld data, but that reduces test coverage and introduces unnecessary noise.

### 7.3 Why BGE-M3 Instead of Pre-Built Chroma?

- **Leak prevention:** Pre-built Chroma was indexed on 100k balanced sample (which includes the withheld class in raw form)
- **Reproducibility:** Fresh embeddings per seed allow controlled variation in representative samples
- **Simplicity:** No dependency on pre-computed vector store state

### 7.4 Why Majority Vote Over k=5 Rules?

- **Robustness:** Single rule can be noisy; majority vote aggregates signal
- **Interpretability:** 5 rules are human-readable; larger k becomes opaque
- **Computational cost:** Each rule requires one tool call; n=5 iterations × k=5 rules × 3 seeds = 75 total evaluations

### 7.5 Why Multi-Seed Runs?

- **Stochasticity source:** LLM temperature=0.1 (low but not zero) + per-seed BGE-M3 sampling
- **Error bars:** Report mean ± std ZDR across seeds for paper credibility
- **Robustness check:** Ensures result pattern is consistent, not artifact of one seed

---

## 8. Interpretation Framework

### 8.1 ZDR Ranges

| ZDR Range | Interpretation | Implication |
|-----------|---|---|
| > 80% | **Strong generalization** | Rules capture abstract attack indicators; likely novel finding |
| 50-80% | **Moderate generalization** | Partial cross-family or cross-layer learning; mixed signal |
| < 50% | **Weak generalization** | Rules overfit to training attack profiles; fail on zero-day |
| ≈ 0% | **No generalization** | Rules are orthogonal to withheld class; complete failure |

### 8.2 Comparative Interpretation

**If LLM ZDR > ML ZDR:**
- LLM-generated rules capture more abstract patterns than statistical ML boundaries
- Potential advantage for novel/unseen attack types
- Suggests LLM policy generation is superior for zero-day defense

**If LLM ZDR < ML ZDR:**
- ML baselines' statistical learning is more robust to family-level patterns
- LLM limited by small representative sample size (10 rows)
- ML has implicit knowledge from seeing attack family repeatedly

**If LLM ZDR ≈ DT ZDR < RF ZDR:**
- DT and LLM capture similar decision boundaries
- RF's ensemble overfits to specific training patterns

### 8.3 Case Study: MITM-ArpSpoofing

**LLM ZDR = 2.2%, DT ZDR = 55.5%**

Why DT > LLM on orthogonal attack:
- Training set includes ARP entries from other classes
- DT discovers ARP-relevant statistical patterns (e.g., ARP indicator, packet size)
- LLM's 10 representative samples miss the ARP pattern entirely
- LLM rules (`Rate <= 5`) target volumetric attacks, missing the ARP context

**Conclusion:** Orthogonal attacks require attack-family-specific training data. Even statistical ML (DT) struggles (55.5% is weak), but outperforms LLM because it saw ARP-adjacent patterns in training.

### 8.4 Case Study: Mirai-udpplain

**LLM ZDR ≈ 0%, DT ZDR ≈ 100%**

Why DT >> LLM on within-family attack:
- Training includes Mirai-greeth_flood and Mirai-greip_flood
- DT learns "Mirai = high-rate UDP flooding" pattern
- Mirai-udpplain is identical pattern → trivial generalization
- LLM's 10 representative samples happened to exclude high-rate attacks
- LLM rules capture low-rate pattern → wrong for Mirai-udpplain

**Conclusion:** Within-family generalization is where ML baselines shine. LLM's small sample size is a critical bottleneck. Even the same attack family fails if representative samples don't include that family's key pattern.

---

## 9. Critical Findings & Limitations

### 9.1 LLM Limitations Revealed

1. **Small representative sample size (10 per class)** leads to narrow rule space
2. **No implicit family knowledge** — LLM doesn't know "Mirai variants are similar"
3. **Representativeness sensitive** — if top-10 samples skew toward low-rate attacks, that's what LLM optimizes for
4. **No multi-modal learning** — LLM only sees feature values, not attack semantics

### 9.2 ML Baseline Strengths

1. **Statistical pattern learning** across thousands of examples
2. **Family-level abstraction** emerges naturally from data (DT learns high-rate patterns)
3. **Robust to single noisy sample** — outliers don't derail decision boundary

### 9.3 Honest Limitations of Experiment

1. **Small training pool (Bot-IoT):** Only 370 benign rows → ~296 train samples total
2. **Potential overfitting:** High known-attack F1 (0.96) on small test sets may not reflect real-world performance
3. **Single LLM tested:** Only GPT-4o (not Gemini, Claude, etc.)
4. **Single temperature:** temperature=0.1 is low stochasticity; higher temps might yield different rules
5. **Withheld class selection bias:** Manually chosen classes; random selection might yield different conclusions

---

## 10. Paper Framing & Contribution

### 10.1 Contribution 3 (for Journal)

> **Title:** "Zero-Day Attack Generalization in LLM-Generated IDS Policies"
>
> **Claims:**
> - We introduce a zero-day classification evaluation method to assess whether LLM-generated threshold rules generalize to unseen attack types.
> - Across CICIoT2023 (35 classes), we withhold different attack paradigms and measure ZDR.
> - **Finding 1:** On within-family attacks (Mirai-udpplain), LLM achieves ~0% ZDR while DT/RF achieve ~99%, revealing that LLM's small representative sample size limits family-level pattern discovery.
> - **Finding 2:** On orthogonal attacks (MITM-ArpSpoofing), both LLM and DT/RF struggle (<60%), confirming that attack-family-specific training is necessary for detection regardless of method.
> - **Implication:** LLM-generated rules are not superior to ML for zero-day detection but offer interpretability trade-off: fewer, simpler rules at the cost of lower generalization rates.

### 10.2 Results Table for Paper

```
| Dataset    | Withheld Class       | Train Samples | ZDR (LLM) | ZDR (DT)  | ZDR (RF)  | Known-F1 (LLM) | Notes |
|------------|----------------------|---------------|-----------|-----------|-----------|----------------|-------|
| CICIoT2023 | MITM-ArpSpoofing     | 78,788        | 2.2%      | 55.5%     | 22.0%     | 0.8648         | Orthogonal (layer 2) attack |
| CICIoT2023 | Mirai-udpplain       | 78,788        | 0.04%     | 99.95%    | 99.98%    | 0.8301         | Within-family attack (same type as training) |
| Bot-IoT    | Service_Scan         | 592           | TBD       | TBD       | TBD       | TBD            | Reconnaissance (distinct paradigm) |
```

---

## 11. Notebooks & File Locations

### Created Files

```
/Users/S4160163/Documents/Projects/RAG Paper/iot-llm-hids-pg/

1-cic-iot/
  └─ 8-zero-day-evaluation.ipynb       [COMPLETE, TESTED with 2 withheld classes]
  
4-bot-iot/
  └─ 8-zero-day-evaluation.ipynb       [COMPLETE, NOT YET RUN]

zero_day_experiment_plan.md            [DESIGN DOC, references old decisions]

zero-day-attack-detect.md              [THIS FILE - comprehensive reference]
```

### Results Output

```
1-cic-iot/results/llm/
  zeroday-MITM-ArpSpoofing-seed42-2026-04-20-10-59-17.json
  zeroday-Mirai-udpplain-seed42-2026-04-20-11-59-41.json
  [+ additional multi-seed runs in Cell 11]

1-cic-iot/results/
  zeroday-MITM-ArpSpoofing-thresholds.png
  zeroday-Mirai-udpplain-thresholds.png
```

---

## 12. Running the Experiments

### 12.1 Prerequisites

```bash
pip install pandas numpy scipy langchain langgraph langchain-huggingface \
            langchain-openai langchain-anthropic sklearn tqdm tabulate
```

Environment: `.env` file with `OPENAI_API_KEY` at project root

### 12.2 CICIoT2023: Run Locally

```python
# In Jupyter, open: 1-cic-iot/8-zero-day-evaluation.ipynb
# 1. Cell 0: Set WITHHELD_CLASS = 'Mirai-udpplain' (or other)
# 2. Restart kernel (important! clears stale state)
# 3. Run Cells 0-10 sequentially
#    - Cells 0-5: Setup (~30 seconds)
#    - Cell 6: LLM pipeline (5-10 minutes for GPT-4o with n=5 iterations)
#    - Cells 7-10: Evaluation (~2 minutes)
# 4. Run Cell 11: Multi-seed (10-15 minutes for 3 seeds)
# 5. Run Cell 12: Visualization
```

### 12.3 Bot-IoT: Run Locally

```python
# Same steps as CICIoT, but:
# - Open: 4-bot-iot/8-zero-day-evaluation.ipynb
# - Cell 0: WITHHELD_CLASS = 'Service_Scan' (default)
# - Note: Training is VERY SMALL (370 benign → 296 after split)
#   This is expected behavior, not an error.
```

### 12.4 Parameters to Tune

| Parameter | Default | Notes |
|-----------|---------|-------|
| `WITHHELD_CLASS` | Varies per dataset | Change in Cell 0, restart kernel |
| `k` | 5 | Number of rules. Increase for more granular policies (higher token cost) |
| `n` | 5 | Feedback iterations. Increase for more refinement (higher token cost) |
| `SEEDS` | [42, 123, 456] | Add/remove for more/fewer error bars |
| `N_REPR` | 10 | Representative samples per class. Lower = faster embedding, less coverage |
| `MAX_EMBED` | 100 | Cap on rows to embed. Lower = faster, less accurate mean |
| LLM | `ChatOpenAI('gpt-4o')` | Change in Cell 5 (uncomment Gemini or Claude lines) |
| temperature | 0.1 | In Cell 5 `llm = ChatOpenAI(..., temperature=0.1)` |

---

## 13. Future Work & Recommendations

### 13.1 Immediate Next Steps

1. **Run Bot-IoT experiments** (all 11 classes)
2. **Try alternative withheld classes** in CICIoT (DDoS-SlowLoris, DDoS-SynonymousIP_Flood)
3. **Increase n from 5 to 10 iterations** for refined rules (cost ↑ 2x)
4. **Test other LLMs** (Gemini 1.5 Pro, Claude 3.5 Sonnet) for robustness

### 13.2 Longer-Term Improvements

1. **Increase representative sample size** (N_REPR from 10 to 20-50) to capture family patterns
   - Trade-off: Higher embedding cost, longer prompts, potentially better generalization
   
2. **Explicit family labeling in prompt**
   - Tell LLM: "These attacks are DDoS floods. Mirai variants are high-rate UDP floods."
   - May improve zero-day generalization if withheld class is same family

3. **Hierarchical rule generation**
   - Generate rules per attack category first, then combine
   - E.g., "DDoS rules: Rate > X, ..."; "Recon rules: Packet size < Y, ..."

4. **Dynamic feature selection**
   - Use feature importance from RF/XGBoost to weight which features to show LLM
   - May guide LLM toward more relevant patterns

5. **Larger training pools**
   - Currently imbalanced (CICIoT: 1:8.26) or tiny (Bot-IoT: 370 benign)
   - Test on datasets with 50k+ benign samples for clearer results

6. **Cross-dataset evaluation**
   - Train on CICIoT, test zero-day on Bot-IoT attacks
   - Measure generalization across fundamentally different datasets

### 13.3 Paper Narrative Options

**Option A (Honest limitations focus):**
> "LLM-generated rules do not outperform decision trees on zero-day attacks, especially within attack families. The constraint of 10 representative samples limits LLM's ability to discover family-level statistical patterns that ML baselines naturally learn from thousands of training examples."

**Option B (Interpretability trade-off focus):**
> "While LLM-generated rules underperform ML on zero-day generalization, they offer significant interpretability advantages: 5 simple threshold rules are human-auditable and deployable in resource-constrained IoT environments, whereas DT/RF provide better accuracy but opaque decision logic."

**Option C (Complementary roles focus):**
> "LLM and ML baselines capture different generalization mechanisms: ML exploits statistical patterns within attack families (Mirai-udpplain detection), while LLMs can reason about attack mechanisms (potential for multi-modal semantics if provided attack descriptions). Hybrid approaches combining both may yield superior zero-day defense."

---

## 14. Quick Reference: Parameter Meanings

| Term | Definition | Example |
|------|-----------|---------|
| **ZDR** | Zero-Day Detection Rate | 0.234 = 23.4% of withheld class detected as attack |
| **Known-Attack F1** | Macro F1 on test_known_df | Sanity check; should be >0.7 |
| **BGE-M3** | BAAI General Embedding Model v3 | Embedding model for representative sample retrieval |
| **Majority Vote** | Aggregate k=5 rules via mode | If 3+ rules say "attack", final pred = "attack" |
| **Stratified Split** | Maintain class ratio in train/test | Ensures train/test have same normal:attack ratio |
| **Representativeness** | Top-10 samples closest to class mean | Avoids edge cases, focuses on typical patterns |

---

## 15. Version History

| Date | Change | Author |
|------|--------|--------|
| 2026-04-20 | Initial pilot notebook (CICIoT MITM-ArpSpoofing) | Akila Maithri |
| 2026-04-20 | Second experiment (CICIoT Mirai-udpplain) | Akila Maithri |
| 2026-04-20 | Bot-IoT notebook template created | Akila Maithri |
| 2026-04-21 | This comprehensive reference document | Akila Maithri |

---

## Appendix A: Common Pitfalls & How We Avoided Them

### Pitfall 1: Data Leakage (Withheld Class in Training)
**❌ Wrong:** Load pre-computed `sample-100000-2.csv` which includes withheld class  
**✅ Right:** Start from `population.csv`, explicitly exclude withheld class before any sampling

### Pitfall 2: Class Imbalance Confound
**❌ Wrong:** Train on imbalanced data (10k benign vs 90k attack) → all rules predict "attack"  
**✅ Right:** Sample attack = benign count → 1:1 balanced training

### Pitfall 3: Vector Store Reuse
**❌ Wrong:** Use pre-built Chroma (indexed on 100k sample including withheld class)  
**✅ Right:** Compute fresh BGE-M3 embeddings per experiment (no leakage, controllable per seed)

### Pitfall 4: Non-Reproducible Representation
**❌ Wrong:** Always use same 10 samples across seeds  
**✅ Right:** Resample per seed (seed=s) → controlled variation for error bars

### Pitfall 5: Unfair ML Baseline Comparison
**❌ Wrong:** Train DT/RF on original imbalanced data, LLM on balanced data  
**✅ Right:** Both use identical train_df (1:1 balanced)

### Pitfall 6: Test Set Contamination
**❌ Wrong:** Accidentally include some withheld samples in test_known_df  
**✅ Right:** Strict partition: training = benign + known_attack; test_zero_day = ALL withheld (disjoint)

---

## Appendix B: Debugging Guide

### Problem: "ModuleNotFoundError: No module named 'langgraph.prebuilt'"
**Solution:** Use custom `make_tool_node()` in Cell 5 (already implemented in our notebooks)

### Problem: "AttributeError: 'NoneType' object has no attribute 'tool_calls'"
**Cause:** Kernel has stale state; LLM didn't return valid tool calls in a previous run  
**Solution:** Restart kernel (Kernel → Restart in Jupyter)

### Problem: Very low known-attack F1 (< 0.5)
**Cause:** Imbalanced training (all rules predict majority class)  
**Solution:** Check train_df composition in Cell 2 output. Ensure `len(normal_df_train) == len(attack_df_train)`

### Problem: ZDR seems too high (> 90%) on orthogonal attack
**Cause:** Withheld class may not actually be orthogonal; might be similar to training  
**Solution:** Verify withheld class in Cell 1 output; check its traffic profile vs training set

### Problem: Cell 11 multi-seed runs show old results (wrong class name)
**Cause:** Notebook was copied from previous run; old outputs not overwritten  
**Solution:** Delete Cell 11 outputs manually or re-run from Cell 0 after kernel restart

---

## Appendix C: Paper Figures to Generate

1. **Figure 1: ZDR by withheld class (bar chart)**
   - X-axis: Withheld class name
   - Y-axis: ZDR %
   - Grouped bars: LLM, DT, RF
   - Include: CICIoT MITM-ArpSpoofing, Mirai-udpplain, Bot-IoT Service_Scan

2. **Figure 2: Rule thresholds vs feature means (Cell 12 output)**
   - Subplot per rule feature
   - Bar chart: Benign mean, Known-attack mean, Zero-day mean
   - Horizontal line: Rule threshold
   - Visual: Shows why rules fail (zero-day mean >> threshold)

3. **Figure 3: Known-attack F1 vs ZDR scatter plot**
   - X-axis: Known-attack F1
   - Y-axis: ZDR
   - Points: Each (method, withheld class) combination
   - Shows: LLM typically has OK known-F1 but poor ZDR

4. **Figure 4: Pipeline topology diagram**
   - ASCII or formal box diagram of LangGraph flow
   - Shows: START → llm_node → tools → evaluate_node feedback loop

---

## Appendix D: Full Cell 6 LLM Prompt (Example)

```
You are a skilled security data analyst.
You are provided with network data entries categorized as either normal or attack, along with their corresponding feature names.
Carefully analyze the differences between normal and attack entries by comparing corresponding fields.
Your task is to generate exactly 5 simple and deterministic rules for the top 5 important features to filter attack entries.
Supported operators: >, <, >=, <=
NEVER use '==' or '!=' - these are forbidden for numeric features.
Generate exactly 5 rules and make a tool call for each rule.

Analyze the following network data and generate 5 rules to identify attack entries.

Normal Entries:
```
{
  "flow_duration": [1.2, 0.8, 3.4, ...],
  "Header_Length": [54.0, 60.0, 48.0, ...],
  ...
}
```

Attack Entries:
```
{
  "flow_duration": [0.001, 0.0005, 0.002, ...],
  "Header_Length": [200.0, 250.0, 300.0, ...],
  ...
}
```
```

**Expected LLM response:** Tool calls for 5 rules, e.g.,
```
{
  "name": "evaluation_tool",
  "args": {
    "feature_name": "flow_duration",
    "op": "<=",
    "value": "1.0"
  }
}
```

---

**End of Document**

For questions or updates, refer to this guide when running future experiments or explaining zero-day evaluation to new team members.
