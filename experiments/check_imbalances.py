#!/usr/bin/env python3
"""Check class imbalance ratios across all datasets."""

import sys
import os
from pathlib import Path

proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))
os.chdir(str(proj_root))

from lib.policy_pipeline.datasets import load_dataset

datasets = ["cic", "wustl", "ton", "bot", "unsw"]

print("=" * 70)
print("CLASS IMBALANCE ANALYSIS (BINARY CLASSIFICATION)")
print("=" * 70)

results = []
for ds_key in datasets:
    try:
        split = load_dataset(ds_key, seed=42)
        n_normal = len(split.normal_train)
        n_attack = len(split.attack_train)
        ratio = n_attack / n_normal if n_normal > 0 else 0
        imbalance_factor = max(n_normal, n_attack) / min(n_normal, n_attack) if min(n_normal, n_attack) > 0 else 0

        results.append({
            "Dataset": ds_key.upper(),
            "Normal": n_normal,
            "Attack": n_attack,
            "Total": n_normal + n_attack,
            "Attack %": f"{(n_attack / (n_normal + n_attack) * 100):.1f}%",
            "Imbalance Factor": f"{imbalance_factor:.1f}:1",
        })
    except Exception as e:
        results.append({
            "Dataset": ds_key.upper(),
            "Normal": "—",
            "Attack": "—",
            "Total": "—",
            "Attack %": "—",
            "Imbalance Factor": f"ERROR: {str(e)[:30]}",
        })

print("\n")
for r in results:
    print(f"{r['Dataset']:10} | Normal: {str(r['Normal']):7} | Attack: {str(r['Attack']):7} | " +
          f"Total: {str(r['Total']):8} | Attack: {str(r['Attack %']):6} | Imbalance: {r['Imbalance Factor']}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("Imbalance Factor > 10:1 → Severe imbalance (hybrid likely beneficial)")
print("Imbalance Factor 2:1 to 10:1 → Moderate imbalance (hybrid uncertain)")
print("Imbalance Factor < 2:1 → Balanced (hybrid likely harmful or no-op)")
