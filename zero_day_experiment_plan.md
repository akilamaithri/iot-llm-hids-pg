# Zero-Day Attack Classification Experiment

## Overview

This experiment tests whether the RAG-LLM policy generation framework can detect **previously unseen attack types** — a realistic IDS deployment scenario where new attack variants emerge after the detection policy was trained. We call the withheld class the "zero-day" attack.

The pipeline itself is unchanged. Only the data preparation changes: one attack class is fully removed from training and then introduced at test time. The key metric is the **Zero-Day Detection Rate (ZDR)**: the fraction of withheld-class entries that the generated rules correctly flag as "attack".

---

## Core Design

```
Training data:  BenignTraffic  +  All attacks EXCEPT withheld class  →  binary (normal / attack)
Test data:      BenignTraffic  +  Known attacks  +  Withheld class (zero-day)
                ↑ standard eval                    ↑ zero-day eval (ZDR)
```

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ZDR** (Zero-Day Detection Rate) | flagged_as_attack / total_withheld | Primary — how many zero-day entries are caught |
| **Known-Attack F1** | standard macro-F1 on test set | Sanity check — should not degrade vs. baseline |
| **Benign FPR** | benign flagged as attack / total benign | False alarm rate |

---

## Class Selection Per Dataset

### CICIoT2023 (34 classes — PILOT)

| Round | Withheld Class | Justification |
|-------|---------------|---------------|
| 1 | **Backdoor_Malware** | Covert, persistence-based attack vs. predominantly volumetric DDoS/DoS/Mirai training set (~80%). Fundamentally different traffic profile: low-rate, stealthy C2 communication vs. high-rate flooding. Most challenging and publishable zero-day candidate. |
| 2 | **SqlInjection** | Application-layer attack (HTTP payload manipulation); training data is almost entirely transport-layer floods. Tests cross-layer generalization — if threshold rules on packet-header features can catch payload-based attacks. |
| 3 | **MITM-ArpSpoofing** | ARP-layer spoofing with unique indicator (ARP=1, low traffic volume). Distinct from IP-layer attacks. ARP indicator is a binary feature the LLM could plausibly pick up. |

**Paper argument for Round 1 (Backdoor_Malware):**
> "We select Backdoor_Malware as the primary zero-day candidate for CICIoT2023 because it represents a stealth/persistence attack paradigm orthogonal to the volumetric flood attacks comprising the majority of training samples. If LLM-generated threshold rules — derived from DDoS/DoS/Mirai patterns — generalise to detect malware backdoor traffic, this constitutes strong evidence that the framework captures abstract attack indicators beyond the specific attack types it was trained on."

### WUSTL-IIoT-2021 (4 classes)

| Round | Withheld Class | Justification |
|-------|---------------|---------------|
| 1 | **Injection** (Command Injection) | Application-layer attack distinct from DoS/Reconnaissance in the remaining training set. IIoT-specific threat vector. Tests whether network-flow rules can detect command injection. |

### TON_IoT (10 classes)

| Round | Withheld Class | Justification |
|-------|---------------|---------------|
| 1 | **ransomware** | Cryptographic/C2 attack with unique traffic behavior — encrypted beaconing to C2 server. High real-world threat relevance (most damaging attack type in enterprise/OT). |
| 2 | **backdoor** | Stealth/persistence (mirrors CICIoT2023 Round 1 for cross-dataset comparison of the same paradigm). |

### Bot-IoT (11 classes)

| Round | Withheld Class | Justification |
|-------|---------------|---------------|
| 1 | **theft** (Information Theft) | Data exfiltration attack — distinct from volumetric DDoS/DoS botnet attacks that dominate this dataset. Tests whether botnet detection rules generalise to data theft traffic. |

### UNSW-NB15 (10 classes)

| Round | Withheld Class | Justification |
|-------|---------------|---------------|
| 1 | **Shellcode** | Low-level exploitation payload — fundamentally different from scanning/DoS/fuzzing in training. Represents a specific post-exploitation technique. |
| 2 | **Worms** | Self-propagating lateral movement — distinct traffic pattern from all other categories. Cross-host propagation not present in other attack types. |

---

## Experimental Protocol

### Step 1: Load multiclass data
- Source: `data/population.csv` (full deduplicated dataset with original labels)
- NOT `sample-100000-2.csv` (that has pre-merged binary labels)

### Step 2: Partition data
```python
benign_df    = df[df['label'] == 'BenignTraffic']
known_atk_df = df[(df['label'] != 'BenignTraffic') & (df['label'] != WITHHELD_CLASS)]
zeroday_df   = df[df['label'] == WITHHELD_CLASS]

# Training pool: benign + known attacks, binary-labelled
train_pool = pd.concat([
    benign_df.sample(n=10800, random_state=seed),
    known_atk_df.sample(n=min(89200, len(known_atk_df)), random_state=seed)
])
train_pool['label'] = train_pool['label'].apply(
    lambda x: 'normal' if x == 'BenignTraffic' else 'attack'
)

# 80/20 split of known data
train_df = train_pool.sample(frac=0.8, random_state=seed)
test_known_df = train_pool.drop(train_df.index)

# Zero-day test set: ALL samples of the withheld class
# (no downsampling — maximises ZDR reliability)
test_zeroday_df = zeroday_df.drop(columns=['label'])
```

### Step 3: Get representative samples (for LLM prompt)
```python
# Feature-space representative samples (fast, no Chroma rebuild needed)
def get_representative_samples(df, n=10):
    """Return n samples closest to the feature-space mean."""
    mean_vec = df.mean(axis=0).values.reshape(1, -1)
    from scipy.spatial.distance import cdist
    distances = cdist(df.values, mean_vec, metric='euclidean').flatten()
    return df.iloc[np.argsort(distances)[:n]]

# Note: For exact replication of original methodology, rebuild Chroma with training data only
# (persist to ./vector-stores/chroma-db-zeroday-{WITHHELD_CLASS}/)
```

### Step 4: Run pipeline (unchanged)
- LangGraph + GPT-4o, temperature=0.1
- k=5 rules, n=5 iterations
- 3 seeds: [42, 123, 456]

### Step 5: Evaluate
```python
# Standard evaluation (known classes only)
evaluate_rules_on_test_set(best_tool_calls, test_known_df)  # → standard F1

# Zero-day evaluation (novel)
def evaluate_zero_day(tool_calls, zeroday_df):
    y_pred = []
    for i in range(len(zeroday_df)):
        votes = [
            "attack" if operators[r['op']](zeroday_df.iloc[i][r['feature_name']], r['value'])
            else "normal"
            for r in tool_calls
        ]
        y_pred.append(mode(votes))
    zdr = sum(p == 'attack' for p in y_pred) / len(y_pred)
    return zdr, y_pred
```

### Step 6: ML baseline comparison
- Train DT + RF on same `train_df`
- Evaluate on `test_known_df` → standard F1 for comparison
- Evaluate on `test_zeroday_df` → DT ZDR, RF ZDR

### Step 7: Results table
| Method | Known-Attack F1 | ZDR | FPR (Benign) |
|--------|----------------|-----|--------------|
| LLM rules | ... | ... | ... |
| Decision Tree | ... | ... | ... |
| Random Forest | ... | ... | ... |

---

## Interpretation Guide

| ZDR | Interpretation |
|-----|---------------|
| > 0.8 | Strong generalization: rules capture abstract attack patterns beyond training attacks |
| 0.5 – 0.8 | Moderate generalization: partial detection of zero-day |
| < 0.5 | Weak generalization: rules overfit to known attack profiles |

**Interesting result patterns:**
- LLM ZDR > DT ZDR: LLM rules are more abstract/generalizable than ML decision boundaries
- LLM ZDR < DT ZDR: DT captures statistical patterns that also apply to zero-day; LLM is more conservative
- High ZDR for Backdoor_Malware despite being very different from training → LLM picks up generic "attack-like" network patterns
- Low ZDR for SqlInjection → application-layer attacks are fundamentally undetectable by packet-header threshold rules

---

## Paper Framing

This experiment adds **Contribution 3** to the journal paper:

> **C3:** "We evaluate the zero-day generalisation capability of LLM-generated IDS policies by withholding attack classes during training and measuring Zero-Day Detection Rates at inference. Across five datasets, we select scientifically defensible zero-day candidates representing attack paradigms orthogonal to the training distribution, and compare generalisation against Decision Tree and Random Forest baselines."

Key claim to test:
> "LLM-generated threshold rules exhibit [higher/comparable/lower] zero-day detection rates than ML decision boundaries, suggesting that LLMs [do/do not] capture abstract attack indicators beyond the specific attack profiles they observe."

---

## Files

| File | Purpose |
|------|---------|
| `1-cic-iot/8-zero-day-evaluation.ipynb` | Pilot notebook (CICIoT2023) |
| `2-wustl-iiot/8-zero-day-evaluation.ipynb` | WUSTL-IIoT replication |
| `3-ton-iot/8-zero-day-evaluation.ipynb` | TON_IoT replication |
| `4-bot-iot/8-zero-day-evaluation.ipynb` | Bot-IoT replication |
| `5-unsw-nb15/8-zero-day-evaluation.ipynb` | UNSW-NB15 replication |
| `zero_day_experiment_plan.md` | This document |
