#!/usr/bin/env python3
"""Run hybrid RAG policy generation on TON-IoT (binary, lazy variant).

Lazy hybrid = sample 1000 flows per class, embed, retrieve top-10 by centroid
similarity for concrete examples. Pair with statistical guidance in prompts.

Expected runtime: ~5-10 minutes (5 rounds + LLM + scoring)
Baseline: 0.7696 → Expected improvement: 0.77-0.82 (+1-5%)
"""

import json
import sys
import os
from pathlib import Path

# Add iot-llm-hids-pg to path
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root))
os.chdir(str(proj_root))

from lib.policy_pipeline.config import RunConfig
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.pipeline_hybrid import run_pipeline_hybrid
from lib.policy_pipeline.hybrid_retrieval import hybrid_retrieve


def main():
    print("[Setup] Loading TON-IoT dataset...")
    split = load_dataset("ton", seed=42)
    normal_train = split.normal_train
    attack_train = split.attack_train

    print(f"  Normal training samples: {len(normal_train)}")
    print(f"  Attack training samples: {len(attack_train)}")
    print(f"  Attack class labels: {split.attack_class_labels_train is not None}")

    print("\n[Hybrid] Sampling and embedding flows...")
    print(f"  Sampling 1000 per class (or all if smaller)")
    print(f"  Embedding model: BAAI/bge-small-en-v1.5")

    normal_examples, attack_examples = hybrid_retrieve(
        normal_train, attack_train,
        seed=42,
        n_samples=1000,
        n_retrieve=10
    )

    print(f"  Retrieved {len(normal_examples)} normal examples")
    print(f"  Retrieved {len(attack_examples)} attack examples")

    # Sample show
    if normal_examples:
        keys = list(normal_examples[0].keys())[:3]
        print(f"  Example normal flow (first 3 features): {keys}")
    if attack_examples:
        keys = list(attack_examples[0].keys())[:3]
        print(f"  Example attack flow (first 3 features): {keys}")

    # Config: match the reference pipeline
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

    print("\n[Pipeline] Starting hybrid policy generation...")
    print(f"  Config: k={cfg.k}, max_rounds={cfg.max_rounds}, voting={cfg.voting_mode}")

    result = run_pipeline_hybrid(
        cfg=cfg,
        normal_train=normal_train,
        attack_train=attack_train,
        attack_class_labels=split.attack_class_labels_train,
        normal_examples=normal_examples,
        attack_examples=attack_examples,
        verbose=True
    )

    # Save results (convert to dict manually)
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / "ton_hybrid_binary_result.json"
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
    print("[Results] Hybrid Policy Generation Complete")
    print("="*60)
    print(f"Best round: {result.best_round_index + 1}")
    print(f"Best macro F1: {result.best_metrics['macro_f1']:.4f}")
    print(f"Attack F1: {result.best_metrics['attack_f1']:.4f}")
    print(f"Attack precision: {result.best_metrics['attack_precision']:.4f}")
    print(f"Attack recall: {result.best_metrics['attack_recall']:.4f}")
    print(f"N rules in best policy: {len(result.best_policy.rules)}")
    print(f"\nBaseline (pure stats RAG): 0.7696 macro F1")
    improvement = result.best_metrics['macro_f1'] - 0.7696
    pct = (improvement / 0.7696) * 100
    print(f"Hybrid result: {result.best_metrics['macro_f1']:.4f} macro F1")
    print(f"Improvement: {improvement:+.4f} ({pct:+.1f}%)")
    print(f"\nSaved to: {result_file}")


if __name__ == "__main__":
    main()
