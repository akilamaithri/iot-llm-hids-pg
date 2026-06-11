#!/usr/bin/env python3
"""Compare hybrid RAG results across datasets."""

import json
from pathlib import Path
from tabulate import tabulate

results_dir = Path("experiments/results")

# Define baselines from project status
BASELINES = {
    "unsw": 0.7040,
    "ton": 0.7696,
    "cic": 0.9426,
    "wustl": 0.9252,
    "bot": 0.9232,
}

print("=" * 80)
print("HYBRID RAG RESULTS COMPARISON")
print("=" * 80)

results_data = []

for dataset_key, baseline_f1 in BASELINES.items():
    result_file = results_dir / f"{dataset_key}_hybrid_binary_result.json"

    if not result_file.exists():
        results_data.append({
            "Dataset": dataset_key.upper(),
            "Baseline F1": f"{baseline_f1:.4f}",
            "Hybrid F1": "—",
            "Improvement": "—",
            "Status": "⏳ Running",
        })
        continue

    try:
        with open(result_file) as f:
            result = json.load(f)

        hybrid_f1 = result['best_metrics']['macro_f1']
        improvement = hybrid_f1 - baseline_f1
        pct = (improvement / baseline_f1) * 100

        status = "✅" if improvement > 0 else "❌"

        results_data.append({
            "Dataset": dataset_key.upper(),
            "Baseline F1": f"{baseline_f1:.4f}",
            "Hybrid F1": f"{hybrid_f1:.4f}",
            "Improvement": f"{improvement:+.4f} ({pct:+.1f}%)",
            "Status": status,
        })
    except Exception as e:
        results_data.append({
            "Dataset": dataset_key.upper(),
            "Baseline F1": f"{baseline_f1:.4f}",
            "Hybrid F1": "—",
            "Improvement": "—",
            "Status": f"❌ Error: {str(e)[:20]}",
        })

# Print table
print("\n" + tabulate(results_data, headers="keys", tablefmt="grid"))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

completed = [r for r in results_data if "✅" in r["Status"]]
print(f"Completed: {len(completed)}/{len(results_data)}")

if completed:
    print("\nSuccessful runs:")
    for r in completed:
        print(f"  • {r['Dataset']:10} {r['Improvement']:20} {r['Status']}")

pending = [r for r in results_data if "⏳" in r["Status"]]
if pending:
    print("\nPending:")
    for r in pending:
        print(f"  • {r['Dataset']}")

print("\n" + "=" * 80)
