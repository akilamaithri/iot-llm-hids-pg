"""Hyperparameter sweep harness.

Reads a YAML grid, runs every Cartesian-product config, writes one row
per run to experiments/sweeps/{dataset}-{timestamp}.csv with policy +
metric columns.

    python experiments/sweep_policy.py --dataset cic --grid configs/cic_grid.yaml
    python experiments/sweep_policy.py --dataset all --grid configs/quick_grid.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.policy_pipeline import RunConfig, run_pipeline
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.eval import classification_metrics
from lib.policy_pipeline.io import save_policy
from lib.policy_pipeline.voting import predict


# Token pricing per million (input, output) USD — for cost column
PRICE_PER_M = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gemini-2.5-flash":          (0.30, 2.50),
}


def _load_yaml(path: str) -> dict:
    """Minimal YAML loader — supports `key: [v1, v2]` lists and scalars only.
    Avoids adding pyyaml as a dep."""
    out: dict = {}
    cur_key: str | None = None
    with open(path) as f:
        for line in f:
            s = line.split("#", 1)[0].rstrip()
            if not s.strip():
                continue
            if ":" in s and not s.startswith(" "):
                k, _, v = s.partition(":")
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    out[k.strip()] = [_coerce(x.strip()) for x in v[1:-1].split(",") if x.strip()]
                elif v:
                    out[k.strip()] = _coerce(v)
                else:
                    cur_key = k.strip()
                    out[cur_key] = []
            elif s.startswith(" ") and cur_key:
                v = s.strip().lstrip("-").strip()
                out[cur_key].append(_coerce(v))
    return out


def _coerce(s: str):
    s = s.strip().strip('"').strip("'")
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower(): return float(s)
        return int(s)
    except ValueError:
        return s


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values_lists)]


def make_config(dataset_key: str, overrides: dict) -> RunConfig:
    defaults = dict(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        seed=42, k=5, max_rounds=4, early_stop_patience=2,
        val_slice_frac=0.20, top_n_divergent=15, temperature=0.1,
        voting_mode="weighted", tau=None, weight_fn="precision",
        div_w_tag=0.35, div_w_disagreement=0.30, div_w_feature=0.20, div_w_threshold=0.15,
        composite_alpha=0.60, selection_metric="macro_f1",
    )
    # If only provider is specified, infer model_id
    if "provider" in overrides and "model_id" not in overrides:
        overrides["model_id"] = (
            "claude-haiku-4-5-20251001" if overrides["provider"] == "anthropic"
            else "gemini-2.5-flash"
        )
    defaults.update(overrides)
    defaults["dataset_key"] = dataset_key
    return RunConfig(**defaults)


def cost_usd(provider_model: str, t_in: int, t_out: int) -> float:
    rate = PRICE_PER_M.get(provider_model, (0.0, 0.0))
    return (t_in / 1e6) * rate[0] + (t_out / 1e6) * rate[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset key (cic|wustl|ton|bot|unsw) or 'all'")
    ap.add_argument("--grid", required=True, help="Path to YAML grid")
    ap.add_argument("--out-dir", default="experiments/sweeps")
    ap.add_argument("--limit", type=int, default=None, help="Max runs (debugging)")
    args = ap.parse_args()

    grid = _load_yaml(args.grid)
    overrides_list = expand_grid(grid)
    if args.limit:
        overrides_list = overrides_list[:args.limit]

    datasets = [args.dataset] if args.dataset != "all" else ["cic", "wustl", "ton", "bot", "unsw"]
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    for ds in datasets:
        out_path = out_dir / f"{ds}-{ts}.csv"
        print(f"\n=== Sweep on {ds}: {len(overrides_list)} configs -> {out_path}")

        # Load data once per dataset
        rows = []
        for i, ov in enumerate(overrides_list):
            cfg = make_config(ds, dict(ov))
            print(f"  [{i+1}/{len(overrides_list)}] {ov}")
            split = load_dataset(ds, seed=cfg.seed)
            t0 = time.time()
            try:
                result = run_pipeline(
                    cfg, split.normal_train, split.attack_train,
                    split.attack_class_labels_train, verbose=False,
                )
                X_te = pd.concat([split.normal_test, split.attack_test], axis=0, ignore_index=True)
                y_te = np.concatenate([
                    np.full(len(split.normal_test), "normal"),
                    np.full(len(split.attack_test), "attack"),
                ])
                metrics = classification_metrics(y_te, predict(result.best_policy, X_te))
                n_tags = len({r.phenomenon_tag for r in result.best_policy.rules})
                policy_path = save_policy(cfg, result)
                wall = time.time() - t0

                row = {**cfg.as_dict(),
                       "best_round": result.best_round_index + 1,
                       "n_distinct_tags": n_tags,
                       "tokens_in": result.total_tokens_in,
                       "tokens_out": result.total_tokens_out,
                       "cost_usd": cost_usd(cfg.model_id, result.total_tokens_in, result.total_tokens_out),
                       "wall_seconds": round(wall, 1),
                       "policy_path": policy_path,
                       **metrics}
                rows.append(row)
                print(f"      f1={metrics['macro_f1']:.4f}  attack_f1={metrics['attack_f1']:.4f}  tags={n_tags}  ${row['cost_usd']:.3f}  {wall:.0f}s")
            except Exception as e:
                print(f"      ERROR: {e}")
                rows.append({**cfg.as_dict(), "error": str(e)})

        # Write CSV (union of all row keys)
        all_keys = sorted({k for r in rows for k in r.keys()})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
