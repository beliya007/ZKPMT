"""
Ablation Study Runner.

For each (model, dataset) task, instantiates all six ablation variants
and measures: init time, single-round prove time, total prove time,
verification time, storage cost, and gate count.

Results are saved to ablation/results/ablation_results.json.

Usage:
    python run_ablation.py              # full run (N_REPEAT=3, 100 rounds)
    python run_ablation.py --fast       # quick check (N_REPEAT=1, 20 rounds)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    MLPSmallApprox, MLPMediumApprox, MLPLargeApprox,
    MODEL_CONFIGS,
)
from data_loader import LOADERS
from ablation_variants import ALL_VARIANTS

# ── Configuration ─────────────────────────────────────────────────────────────
N_REPEAT    = 3
NUM_ROUNDS  = 100
BATCH_SIZE  = 32
LR          = 0.01

TASKS = [
    ("Small+Iris",     "Iris"),
    ("Medium+MNIST",   "MNIST"),
    ("Large+CIFAR-10", "CIFAR-10"),
    ("ResNet50+CIFAR-10", "CIFAR-10"),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_task(model_key, dataset_name, num_rounds, n_repeat):
    print(f"\n{'='*60}")
    print(f"  Ablation Task: MLP-{model_key}  |  {dataset_name}")
    print(f"{'='*60}")

    _, approx_cls, input_dim, num_classes, _ = MODEL_CONFIGS[model_key]
    loader = LOADERS[dataset_name]
    dataset = loader()
    X_tr, y_tr = dataset[0], dataset[1]

    aggregated = {}

    for rep in range(n_repeat):
        print(f"  Repetition {rep+1}/{n_repeat}")
        for VariantClass in ALL_VARIANTS:
            name = VariantClass.name
            model = approx_cls(input_dim, num_classes)
            variant = VariantClass(model, (X_tr, y_tr),
                                   lr=LR, batch_size=BATCH_SIZE,
                                   num_rounds=num_rounds)
            print(f"    {name:<22} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            variant.setup()
            result = variant.run()
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")

            if name not in aggregated:
                aggregated[name] = []
            aggregated[name].append(result)

    # Average over repetitions
    averaged = {}
    for name, results in aggregated.items():
        numeric_keys = [k for k, v in results[0].items()
                        if isinstance(v, (int, float))]
        averaged[name] = {"variant": name}
        for k in numeric_keys:
            vals = [r[k] for r in results]
            averaged[name][k] = float(np.mean(vals))
        # Preserve boolean flags from first result
        for k, v in results[0].items():
            if isinstance(v, bool):
                averaged[name][k] = v

    return averaged


def run_all(num_rounds, n_repeat):
    all_results = {}
    for model_key, dataset_name in TASKS:
        label = f"MLP-{model_key}" if "ResNet" not in model_key else model_key
        results = run_task(model_key, dataset_name, num_rounds, n_repeat)
        all_results[label] = results

        out = os.path.join(RESULTS_DIR, "ablation_results.json")
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved: {out}")

    _print_summary(all_results)
    return all_results


def _print_summary(all_results):
    metrics = [
        ("init_time_s",         "Init (s)"),
        ("single_prove_time_s", "Single prove (s)"),
        ("total_prove_time_s",  "Total prove (s)"),
        ("verify_time_s",       "Verify (s)"),
        ("storage_mb",          "Storage (MB)"),
        ("num_gates",           "Gate count"),
    ]
    variants = [v.name for v in ALL_VARIANTS]

    for task, results in all_results.items():
        print(f"\n── {task} ──")
        header = f"{'Metric':<22}" + "".join(f"{v:>18}" for v in variants)
        print(header)
        print("-" * len(header))
        for key, label in metrics:
            row = f"{label:<22}"
            for v in variants:
                val = results.get(v, {}).get(key, float("nan"))
                row += f"{val:>18.2f}"
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    n_repeat   = 1   if args.fast else N_REPEAT
    num_rounds = 20  if args.fast else NUM_ROUNDS
    return run_all(num_rounds, n_repeat)


if __name__ == "__main__":
    main()
