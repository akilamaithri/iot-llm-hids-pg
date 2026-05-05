# Feature Importance Exports — In Progress

Background processes are currently computing ML feature importance files for WUSTL, TON, Bot, UNSW datasets.

## Status (as of 2026-05-05 18:21)

### Completed
- ✅ DT gini importance (5/5 datasets)
- ✅ Attack precision reports (5/5 datasets) — committed to git

### In Progress (background processes)
- ⏳ RF gini importance — WUSTL done, TON/Bot/UNSW computing
- ⏳ Permutation importance — WUSTL done, TON/Bot/UNSW computing

Permutation importance is slow because it fits 5 models (LR, DT, RF, SVM, KNN) with cross-validation on each test set. Estimated completion: 2-4 hours from start time (18:13).

## Files Being Generated

For each of WUSTL, TON, Bot, UNSW:
- `{dataset}/results/feature-importance-100000-dt.txt` — top-5 features (gini)
- `{dataset}/results/feature-importance-100000-rf.txt` — top-5 features (gini)
- `{dataset}/results/feature-importance-100000-permutation.csv` — full importance matrix (5 models × normalized)

## To Commit When Complete

```bash
cd /Users/S4160163/Documents/Projects/RAG\ Paper/iot-llm-hids-pg

# Verify all files exist
ls [2-5]*-*/results/feature-importance-100000-{dt.txt,rf.txt,permutation.csv}

# Stage and commit
git add [2-5]*-*/results/feature-importance-*
git commit -m "Add ML feature importance exports (permutation, DT/RF gini) for WUSTL, TON, Bot, UNSW"
```

## Script Reference

`experiments/12-standardize-results.py` can be rerun to regenerate any dataset:
```bash
python experiments/12-standardize-results.py --dataset all --skip-missing
```
