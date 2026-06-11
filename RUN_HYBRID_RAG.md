# How to Run Lazy Hybrid RAG

## Quick Start (One Command)

```bash
cd iot-llm-hids-pg
python3 experiments/run_unsw_hybrid.py
```

**Expected output:**
```
[Setup] Loading UNSW-NB15 dataset...
[Hybrid] Sampling and embedding flows...
[Pipeline] Starting hybrid policy generation...
=== Round 1/5 ===
...
[Results] Hybrid Policy Generation Complete
============================================================
Best macro F1: 0.8478
Improvement: +0.1438 (+20.4%)
Saved to: experiments/results/unsw_hybrid_binary_result.json
```

**Runtime:** ~5–11 minutes

---

## Verify Installation

### Check Python version and packages

```bash
python3 --version  # Should be 3.9+
pip list | grep -E "langchain|pandas|numpy|huggingface"
```

### Quick test of hybrid_retrieve()

```bash
cd iot-llm-hids-pg
python3 << 'EOF'
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.hybrid_retrieval import hybrid_retrieve

split = load_dataset("unsw", seed=42)
normal_ex, attack_ex = hybrid_retrieve(
    split.normal_train, split.attack_train,
    seed=42, n_samples=1000, n_retrieve=10
)

print(f"✓ Retrieved {len(normal_ex)} normal examples")
print(f"✓ Retrieved {len(attack_ex)} attack examples")
print(f"✓ hybrid_retrieve() works!")
EOF
```

---

## Results Location

After running, check:

```bash
cat experiments/results/unsw_hybrid_binary_result.json | python3 -m json.tool
```

**Key fields:**
```json
{
  "best_metrics": {
    "macro_f1": 0.8478,
    "attack_f1": 0.8705,
    "attack_precision": 0.7707,
    "attack_recall": 1.0
  },
  "best_round_index": 0,
  "best_policy": {
    "rules": [...],  // 5 rules
    "voting_mode": "weighted",
    "tau": 0.7727
  }
}
```

---

## Comparison with Baseline

**Baseline (pure stats RAG):**
```
UNSW-NB15 macro F1 = 0.7040
```

**Hybrid RAG:**
```
UNSW-NB15 macro F1 = 0.8478
Improvement = +0.1438 (+20.4%)
```

---

## Customize: Run on Different Dataset

### Option 1: Create variant for TON-IoT

```bash
cp experiments/run_unsw_hybrid.py experiments/run_ton_hybrid.py
# Edit: change "unsw" → "ton" in run_ton_hybrid.py
python3 experiments/run_ton_hybrid.py
```

### Option 2: Modify config

Edit `experiments/run_unsw_hybrid.py`, line ~60:

```python
cfg = RunConfig(
    dataset_key="ton",  # Change this
    provider="anthropic",
    model_id="claude-haiku-4-5-20251001",
    k=5, max_rounds=5, early_stop_patience=2,
    val_slice_frac=0.15, voting_mode="weighted",
    selection_metric="macro_f1", temperature=0.1,
)
```

Then run:
```bash
python3 experiments/run_unsw_hybrid.py
```

---

## Programmatic Usage

```python
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.pipeline_hybrid import run_pipeline_hybrid
from lib.policy_pipeline.hybrid_retrieval import hybrid_retrieve
from lib.policy_pipeline.config import RunConfig
import json

# 1. Load dataset
split = load_dataset("unsw", seed=42)

# 2. Retrieve examples
normal_examples, attack_examples = hybrid_retrieve(
    split.normal_train, split.attack_train,
    seed=42, n_samples=1000, n_retrieve=10
)

# 3. Configure
cfg = RunConfig(
    dataset_key="unsw",
    provider="anthropic",
    model_id="claude-haiku-4-5-20251001",
    seed=42, k=5, max_rounds=5, early_stop_patience=2,
    val_slice_frac=0.15, voting_mode="weighted",
    selection_metric="macro_f1", temperature=0.1,
)

# 4. Run hybrid pipeline
result = run_pipeline_hybrid(
    cfg=cfg,
    normal_train=split.normal_train,
    attack_train=split.attack_train,
    attack_class_labels=split.attack_class_labels_train,
    normal_examples=normal_examples,
    attack_examples=attack_examples,
    verbose=True
)

# 5. Extract results
print(f"Best macro F1: {result.best_metrics['macro_f1']:.4f}")
print(f"Attack F1: {result.best_metrics['attack_f1']:.4f}")
print(f"Best round: {result.best_round_index}")
print(f"N rules: {len(result.best_policy.rules)}")
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'lib'"

**Fix:** Make sure you're in the correct directory:
```bash
cd iot-llm-hids-pg
python3 experiments/run_unsw_hybrid.py
```

### "HuggingFace embeddings downloading..."

**Normal:** First run downloads the embedding model (~1GB). Subsequent runs use cached version.

**To use cached version only:**
```python
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

### "ANTHROPIC_API_KEY not found"

**Fix:** Set environment variable:
```bash
export ANTHROPIC_API_KEY="sk-..."
python3 experiments/run_unsw_hybrid.py
```

Or create `.env` file in repo root:
```
ANTHROPIC_API_KEY=sk-...
```

### "Timeout / Taking too long"

**Expected timing:**
- Embedding (first run): ~30 sec
- LLM pipeline (5 rounds): ~5–10 min
- **Total: ~5–11 minutes**

If longer:
- Check network (LLM calls might be throttled)
- Try reducing `k=3` or `max_rounds=3` for faster iteration

---

## Output Format

### Console Output

```
[Setup] Loading UNSW-NB15 dataset...
  Normal training samples: 50000
  Attack training samples: 50000
  Attack class labels: True

[Hybrid] Sampling and embedding flows...
  Sampling 1000 per class (or all if smaller)
  Embedding model: BAAI/bge-small-en-v1.5
  Retrieved 10 normal examples
  Retrieved 10 attack examples

[Pipeline] Starting hybrid policy generation...
  Config: k=5, max_rounds=5, voting=weighted

=== Round 1/5 ===
LLM emitted 5 proposals (tokens in/out: 7176/1083)
[HEADER   ] sttl >= 250
[STATE    ] ct_state_ttl >= 1
[VOLUME   ] rate > 50000
[HANDSHAKE] swin == 0
[ASYMMETRY] ct_dst_sport_ltm > 5
Accepted 5 rules, 5 distinct tags
Val macro_f1: 0.8478  (macro_f1=0.8478, attack_f1=0.8705, prec=0.7707, rec=1.0000)

=== Round 2/5 ===
...

============================================================
[Results] Hybrid Policy Generation Complete
============================================================
Best round: 0
Best macro F1: 0.8478
Attack F1: 0.8705
Attack precision: 0.7707
Attack recall: 1.0000
N rules in best policy: 5
Baseline (pure stats RAG): 0.7040 macro F1
Hybrid result: 0.8478 macro F1
Improvement: +0.1438 (+13.0%)
Saved to: experiments/results/unsw_hybrid_binary_result.json
```

### JSON Output

File: `experiments/results/unsw_hybrid_binary_result.json` (2.7 KB)

Contains:
- `config`: RunConfig settings used
- `best_round_index`: Which round was best (0 = Round 1)
- `best_metrics`: Macro F1, Attack F1, Precision, Recall, etc.
- `best_policy`: Rules, voting mode, tau, weights
- `total_tokens_in/out`: LLM token usage

---

## Next Steps

1. **Verify results match (+20.4%)** on your machine
2. **Test on other datasets** (TON-IoT, Bot-IoT)
3. **Optional: Run zero-day evaluation** with hybrid-generated rules
4. **Optional: Integrate into paper** as optional augmentation

---

## Questions?

See `HYBRID_RAG_CHECKLIST.md` for comprehensive FAQ and troubleshooting.

