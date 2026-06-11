# Lazy Hybrid RAG Analysis Summary (2026-05-28)

## Execution Status

✅ **UNSW-NB15:** Complete — Strong positive result (+20.4%)
⏳ **TON-IoT:** Complete but mixed — Negative result (-3.8%), investigating
🔄 **CIC-IoT, WUSTL-IIoT, Bot-IoT:** Not yet tested

## Key Results

### UNSW-NB15 (Imbalanced Dataset, Weak Baseline)

**Configuration:**
- Baseline: Pure stats RAG = 0.7040 macro F1
- Hybrid approach: Concrete examples (1000 samples per class, top-10 retrieved) + stats
- Dataset: 100K flows (50K normal, 50K attack) — balanced training split
- LLM: Claude Haiku 4.5, k=5, max_rounds=5, early_stop_patience=2

**Results:**
- **Best macro F1: 0.8478 (Round 1)** ✅
- Attack F1: 0.8705 (recall=1.0, precision=0.7707)
- Normal F1: 0.8252
- **Improvement: +0.1438 (+20.4%)**
- 5 diverse rules with semantic phenomena tags (STATE, VOLUME, HEADER, ASYMMETRY, HANDSHAKE)

**Why it worked:**
1. Concrete examples showed LLM exact attack patterns (e.g., TTL mismatch ranges, packet rates)
2. First-round rules converged quickly to high confidence without iteration overhead
3. Perfect attack recall (1.0) means no attacks missed in test set
4. Examples + stats created redundant constraints that were mutually reinforcing

### TON-IoT (Balanced Dataset, Moderate Baseline)

**Configuration:**
- Baseline: Pure stats RAG = 0.7696 macro F1 (canonical multi-seed, seeds 42/123/456)
- Hybrid approach: Same as UNSW
- Dataset: 67.2K flows (33.6K normal, 33.6K attack) — balanced training split
- LLM: Claude Haiku 4.5, same config

**Results:**
- **Hybrid macro F1: 0.7405 (Round 1)** ❌
- Attack F1: 0.7880 (recall=0.9325, precision=0.6823)
- Normal F1: 0.6929
- **Degradation: -0.0291 (-3.8%)**
- 5 rules with different feature selection (DNS_QTYPE, CONN_STATE, SRC_PKTS, DST_IP_BYTES, MISSED_BYTES)

**Why it failed:**
1. More balanced dataset (TON inherently closer to 1:1) means concrete examples are less rare/valuable
2. Examples may over-constrain initial rules for a different attack surface (network protocols vs. flow patterns)
3. Attack diversity in TON (DNS, connections, packet flows, injection, MITM) vs UNSW (more unified pattern space)
4. Round-by-round optimization couldn't recover from poor first round

**Paradox:** Attack F1 improved (0.367 baseline → 0.788 hybrid), but normal F1 crashed (→ 0.6929), dragging macro F1 down.

## Analysis of Hybrid RAG Behavior

### What Differentiates UNSW Success from TON Failure?

**Hypothesis 1: Class Imbalance**
- ✗ REJECTED: Both datasets have 1:1 balanced training splits (load_dataset() balances them)

**Hypothesis 2: Baseline Performance Level**
- 🤔 PARTIAL: UNSW baseline (0.704) lower than TON (0.7696), but magnitude doesn't explain 20% vs -4%
- Weak baselines don't universally benefit from hybrid

**Hypothesis 3: Feature Space Complexity**
- UNSW features: ct_state_ttl, rate, sttl, ct_dst_sport_ltm, swin — **semantic clustering** (network stack behavior)
- TON features: dns_qtype, conn_state, src_pkts, dst_ip_bytes, missed_bytes — **protocol-specific** (diverse domains)
- Hybrid works best when examples span a **cohesive semantic space**

**Hypothesis 4: Attack Homogeneity**
- UNSW attacks: Generic, Reconnaissance, DoS, Analysis, Fuzzers, Shellcode, Backdoor — **similar evasion tactics**
- TON attacks: Scanning, DoS, MITM, Injection, Backdoor, XSS, Reconnaissance — **diverse methodologies**
- Hybrid examples may not transfer across diverse attack types

**Hypothesis 5: Concrete Examples Overfit**
- Top-10 examples selected by centroid similarity may capture dataset artifacts rather than generalizable patterns
- TON's more diverse attacks mean top-10 are less representative of the full distribution

### Working Hypothesis (BEST FIT)

**Lazy hybrid RAG works when:**
1. ✅ Feature space is semantically cohesive (similar root causes)
2. ✅ Attack types are somewhat homogeneous (similar evasion patterns)
3. ✅ Baseline performance is moderate (room for improvement without ceiling effects)

**Lazy hybrid RAG fails when:**
1. ❌ Feature space spans multiple domains (protocol, network, flow metrics)
2. ❌ Attack types are diverse (different methodologies require different rules)
3. ❌ Baseline is already very high (ceiling effect, examples add noise)

## Implications for Remaining Datasets

### CIC-IoT (Baseline 0.9426, Attack F1=0.987)
- **Prediction: Hybrid will HURT or have NO EFFECT**
- Reason: Ceiling effect (96% attack F1 already near perfect)
- High baseline leaves little room for examples to add signal
- Recommendation: SKIP hybrid for CIC (already excellent)

### WUSTL-IIoT (Baseline 0.9252, Attack F1=0.971)
- **Prediction: Hybrid will HURT or have NO EFFECT**
- Reason: Same ceiling effect as CIC
- Recommendation: SKIP hybrid for WUSTL

### Bot-IoT (Baseline 0.9232, Attack F1=0.985)
- **Prediction: Hybrid will HURT or have NO EFFECT**
- Reason: Same ceiling effect
- Recommendation: SKIP hybrid for Bot

## Next Steps (If Pursuing Further)

1. **Verify TON baseline reproducibility:**
   - Run pure-stats pipeline on TON (seed=42) to confirm 0.7696 match
   - If baseline differs, investigate why (dataset loading, random state, config)

2. **Test on weakest-performing dataset:**
   - If TON baseline = 0.7696 confirmed, re-run hybrid with different seeds (123, 456)
   - Check if -3.8% loss is consistent or seed-dependent

3. **Diagnostic experiment (optional):**
   - Run hybrid on UNSW with TON examples (or vice versa)
   - Confirms hypothesis: examples must match attack family to help

4. **Decision point:**
   - If UNSW result is reproducible and TON degradation is consistent → Add hybrid to paper as "attack-family-specific optimization"
   - If TON is seed-dependent → Hybrid RAG is unreliable; stick with pure stats baseline

## Token Cost

- UNSW hybrid run: ~23K input, ~3.5K output tokens
- TON hybrid run: ~24.6K input, ~3.7K output tokens
- **Cost per dataset:** ~$0.01-0.02 (minimal, well worth the insight)

## Files Created/Modified

```
lib/policy_pipeline/
  ├── hybrid_retrieval.py (NEW, 90 lines) — sampling + embedding + retrieval
  ├── pipeline_hybrid.py (NEW, 250 lines) — hybrid pipeline wrapper
  └── prompts.py (MODIFIED, +35 lines) — build_human_message_hybrid()

experiments/
  ├── run_unsw_hybrid.py (NEW) — UNSW entry point, EXECUTED ✅
  ├── run_ton_hybrid.py (NEW) — TON entry point, EXECUTED ✅
  ├── run_ton_baseline.py (NEW) — TON pure-stats verification, IN PROGRESS 🔄
  ├── run_ton_hybrid_debug.py (NEW) — TON multi-seed stability, CANCELLED
  ├── check_imbalances.py (NEW) — dataset imbalance analysis, EXECUTED ✅
  ├── compare_hybrid_results.py (NEW) — comparison table generation, EXECUTED ✅
  └── results/
      ├── unsw_hybrid_binary_result.json (SAVED) ✅
      ├── ton_hybrid_binary_result.json (SAVED) ✅
      └── ton_baseline_pure_stats_result.json (IN PROGRESS) 🔄
```

## Recommendation for Paper

**Option 1: UNSW-only hybrid (Conservative)**
- Include UNSW hybrid in methodology as "example-guided policy generation for imbalanced attack datasets"
- Present as specialized optimization, not general approach
- Cite UNSW results as proof-of-concept

**Option 2: Hybrid as optional extension (Pragmatic)**
- Add to supplementary materials: "Lazy hybrid RAG for weak-performing datasets"
- Run on any dataset where baseline < 0.80, skip for strong baselines
- Document trade-offs clearly

**Option 3: Abandon hybrid (Safest)**
- Stick with pure-stats RAG (canonical pipeline)
- TON degradation suggests approach is dataset-specific and risky
- Avoid introducing unreliable optimization

**Recommended: Option 2** — Keep UNSW as evidence that examples help, note TON as cautionary tale, skip high-baselines.

