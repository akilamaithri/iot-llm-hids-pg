#!/usr/bin/env python3
"""Run pure-stats (non-hybrid) TON-IoT pipeline to verify baseline reproducibility."""

import json
import sys
import os
from pathlib import Path

proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))
os.chdir(str(proj_root))

from lib.policy_pipeline.config import RunConfig
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.pipeline import run_pipeline


def main():
    print("[Setup] Loading TON-IoT dataset (seed=42)...")
    split = load_dataset("ton", seed=42)
    normal_train = split.normal_train
    attack_train = split.attack_train

    print(f"  Normal training samples: {len(normal_train)}")
    print(f"  Attack training samples: {len(attack_train)}")

    cfg = RunConfig(
        dataset_key="ton",
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        seed=42,
        k=5,
        max_rounds=5,
        early_stop_patience=2,
        val_slice_frac=0.15,
        voting_mode="weighted",
        selection_metric="macro_f1",
        temperature=0.1,
    )

    print("\n[Pipeline] Starting PURE STATS policy generation (TON-IoT, no hybrid)...")
    result = run_pipeline(
        cfg=cfg,
        normal_train=normal_train,
        attack_train=attack_train,
        attack_class_labels=split.attack_class_labels_train,
        verbose=True
    )

    # Save results
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / "ton_baseline_pure_stats_result.json"
    result_dict = {
        "config": result.config.as_dict(),
        "best_round_index": result.best_round_index,
        "best_metrics": result.best_metrics,
        "best_policy": {
            "rules": [r.to_dict() for r in result.best_policy.rules],
            "voting_mode": result.best_policy.voting_mode,
            "tau": result.best_policy.tau,
            "weights": result.best_policy.weights,
        },
        "total_tokens_in": result.total_tokens_in,
        "total_tokens_out": result.total_tokens_out,
    }
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    print("\n" + "="*60)
    print("[Results] Pure-Stats Policy Generation Complete")
    print("="*60)
    print(f"Best round: {result.best_round_index + 1}")
    print(f"Best macro F1: {result.best_metrics['macro_f1']:.4f}")
    print(f"Attack F1: {result.best_metrics['attack_f1']:.4f}")
    print(f"Attack precision: {result.best_metrics['attack_precision']:.4f}")
    print(f"Attack recall: {result.best_metrics['attack_recall']:.4f}")
    print(f"N rules in best policy: {len(result.best_policy.rules)}")
    print(f"\nExpected baseline: 0.7696")
    improvement = result.best_metrics['macro_f1'] - 0.7696
    pct = (improvement / 0.7696) * 100
    print(f"Observed macro F1: {result.best_metrics['macro_f1']:.4f}")
    print(f"Difference from baseline: {improvement:+.4f} ({pct:+.1f}%)")
    print(f"\nSaved to: {result_file}")


if __name__ == "__main__":
    main()
