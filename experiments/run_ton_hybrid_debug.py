#!/usr/bin/env python3
"""Debug TON-IoT hybrid: re-run with different seeds to check stability."""

import json
import sys
import os
from pathlib import Path

proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))
os.chdir(str(proj_root))

from lib.policy_pipeline.config import RunConfig
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.pipeline_hybrid import run_pipeline_hybrid
from lib.policy_pipeline.hybrid_retrieval import hybrid_retrieve


def run_ton_with_seed(seed: int):
    """Run TON hybrid with a specific seed."""
    print(f"\n{'='*60}")
    print(f"TON-IoT Hybrid Debug Run (seed={seed})")
    print(f"{'='*60}")

    # Load dataset
    split = load_dataset("ton", seed=seed)
    normal_train = split.normal_train
    attack_train = split.attack_train

    print(f"[Data] Normal: {len(normal_train)}, Attack: {len(attack_train)}")

    # Retrieve examples
    normal_examples, attack_examples = hybrid_retrieve(
        normal_train, attack_train,
        seed=seed,
        n_samples=1000,
        n_retrieve=10
    )
    print(f"[Hybrid] Retrieved {len(normal_examples)} normal + {len(attack_examples)} attack examples")

    # Config
    cfg = RunConfig(
        dataset_key="ton",
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        seed=seed,
        k=5,
        max_rounds=5,
        early_stop_patience=2,
        val_slice_frac=0.15,
        voting_mode="weighted",
        selection_metric="macro_f1",
        temperature=0.1,
    )

    # Run
    result = run_pipeline_hybrid(
        cfg=cfg,
        normal_train=normal_train,
        attack_train=attack_train,
        attack_class_labels=split.attack_class_labels_train,
        normal_examples=normal_examples,
        attack_examples=attack_examples,
        verbose=False  # Reduce output
    )

    # Report
    improvement = result.best_metrics['macro_f1'] - 0.7696
    pct = (improvement / 0.7696) * 100

    print(f"[Results]")
    print(f"  Best macro F1: {result.best_metrics['macro_f1']:.4f}")
    print(f"  vs Baseline:   0.7696")
    print(f"  Improvement:   {improvement:+.4f} ({pct:+.1f}%)")
    print(f"  Attack F1:     {result.best_metrics['attack_f1']:.4f}")
    print(f"  Attack recall: {result.best_metrics['attack_recall']:.4f}")

    return result.best_metrics['macro_f1']


if __name__ == "__main__":
    print("Running TON-IoT hybrid with multiple seeds to check stability...")

    results = {}
    for seed in [42, 123, 456]:
        try:
            f1 = run_ton_with_seed(seed)
            results[seed] = f1
        except Exception as e:
            print(f"ERROR (seed={seed}): {e}")
            results[seed] = None

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for seed, f1 in results.items():
        if f1 is not None:
            improvement = f1 - 0.7696
            pct = (improvement / 0.7696) * 100
            print(f"Seed {seed:3d}: F1={f1:.4f}  ({improvement:+.4f}, {pct:+.1f}%)")
        else:
            print(f"Seed {seed:3d}: ERROR")

    # Check if results are stable
    valid_results = [f for f in results.values() if f is not None]
    if len(valid_results) > 1:
        min_f1 = min(valid_results)
        max_f1 = max(valid_results)
        spread = max_f1 - min_f1
        print(f"\nSpread: {spread:.4f} (min={min_f1:.4f}, max={max_f1:.4f})")
        if spread > 0.05:
            print("⚠️  High variance across seeds — results are unstable")
        else:
            print("✓ Stable across seeds")
