"""Aggregate sweep CSVs into a per-(dataset, provider) winner table.

    python experiments/build_sweep_summary.py
    python experiments/build_sweep_summary.py --by attack_f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", default="macro_f1",
                    choices=["macro_f1", "attack_f1", "attack_precision",
                             "attack_recall", "attack_precision_at_recall_0.9"])
    ap.add_argument("--sweeps-dir", default="experiments/sweeps")
    args = ap.parse_args()

    sweeps = list((REPO / args.sweeps_dir).glob("*.csv"))
    if not sweeps:
        print("No sweep CSVs found.")
        return
    print(f"Loading {len(sweeps)} sweep CSV(s)")
    df = pd.concat([pd.read_csv(p) for p in sweeps], ignore_index=True)
    df = df[df.get("error", pd.Series([None]*len(df))).isna()]

    # Group by dataset_key + provider; pick winning row by --by metric
    group_cols = ["dataset_key", "provider"]
    winner_rows = []
    for (ds, prov), g in df.groupby(group_cols):
        best = g.loc[g[args.by].idxmax()]
        winner_rows.append({
            "dataset": ds, "provider": prov,
            "winning_metric": args.by,
            "winning_value": float(best[args.by]),
            "macro_f1": float(best.get("macro_f1", float("nan"))),
            "attack_f1": float(best.get("attack_f1", float("nan"))),
            "attack_precision": float(best.get("attack_precision", float("nan"))),
            "attack_recall": float(best.get("attack_recall", float("nan"))),
            "n_distinct_tags": int(best.get("n_distinct_tags", 0)),
            "voting_mode": best.get("voting_mode", ""),
            "selection_metric": best.get("selection_metric", ""),
            "composite_alpha": float(best.get("composite_alpha", float("nan")))
                                if "composite_alpha" in best else None,
            "k": int(best.get("k", 5)) if "k" in best else 5,
            "tokens_in": int(best.get("tokens_in", 0)),
            "tokens_out": int(best.get("tokens_out", 0)),
            "policy_path": best.get("policy_path", ""),
        })

    summary = pd.DataFrame(winner_rows).sort_values(["dataset", "provider"])
    print("\n=== WINNERS by " + args.by + " ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out = REPO / args.sweeps_dir / f"winners-{args.by}.csv"
    summary.to_csv(out, index=False, float_format="%.4f")
    print(f"\nWrote: {out}")

    # Also emit a Python snippet to paste into multi_seed.WINNERS
    print("\n=== Paste into experiments/multi_seed.py WINNERS dict ===\n")
    for _, r in summary.iterrows():
        extras = {
            "voting_mode": r["voting_mode"],
            "selection_metric": r["selection_metric"],
        }
        if not pd.isna(r["composite_alpha"]):
            extras["composite_alpha"] = float(r["composite_alpha"])
        extras["k"] = int(r["k"])
        print(f'WINNERS[("{r["dataset"]}", "{r["provider"]}")] = {extras}')


if __name__ == "__main__":
    main()
