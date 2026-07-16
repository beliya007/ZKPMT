"""
Experiment runner for zkPMT vs. baselines.

Reproduces the four evaluation dimensions from the paper:
  Fig. 4 – Initialization cost
  Fig. 5 – Single-round proof generation cost
  Fig. 6 – Cumulative proof generation cost (multiple rounds)
  Fig. 7 – Verification cost
  Fig. 8 – Storage cost

Each experiment is repeated N_REPEAT times and the mean is reported.
Results are saved to results/results.json for plotting.
"""

import os
import json
import copy
import time
import numpy as np
import torch

from models import (
    MLPSmall, MLPMedium, MLPLarge,
    MLPSmallApprox, MLPMediumApprox, MLPLargeApprox,
    MODEL_CONFIGS,
)
from data_loader import LOADERS
from zkpmt import zkPMT
from baselines import GargScheme, KaizenScheme, zkCNNScheme, zkDLScheme, VeriCNNScheme

# ── Experiment configuration ─────────────────────────────────────────────────
N_REPEAT    = 3       # repetitions per experiment (paper uses 50; reduce for speed)
NUM_ROUNDS  = 100     # training rounds per run
BATCH_SIZE  = 32
LR          = 0.01

TASKS = [
    ("Small+Iris",     "Iris"),
    ("Medium+MNIST",   "MNIST"),
    ("Large+CIFAR-10", "CIFAR-10"),
    ("ResNet50+CIFAR-10", "CIFAR-10"),
]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_schemes(model_key, dataset):
    """
    Instantiate all six schemes for a given model/dataset combination.
    Returns a list of (scheme_name, scheme_instance) tuples.
    """
    std_cls, approx_cls, input_dim, num_classes, _ = MODEL_CONFIGS[model_key]
    X_tr, y_tr, _, _ = dataset

    schemes = []

    # zkPMT uses the approximated model
    zkpmt_model = approx_cls(input_dim, num_classes)
    zkpmt = zkPMT(zkpmt_model, (X_tr, y_tr),
                  lr=LR, batch_size=BATCH_SIZE, num_rounds=NUM_ROUNDS)
    schemes.append(("zkPMT", zkpmt))

    # All baselines use the standard model
    for SchemeClass, name in [
        (GargScheme,   "Garg"),
        (KaizenScheme, "Kaizen"),
        (zkCNNScheme,  "zkCNN"),
        (zkDLScheme,   "zkDL"),
        (VeriCNNScheme,"VeriCNN"),
    ]:
        m = std_cls(input_dim, num_classes)
        s = SchemeClass(m, (X_tr, y_tr),
                        lr=LR, batch_size=BATCH_SIZE, num_rounds=NUM_ROUNDS)
        schemes.append((name, s))

    return schemes


def run_single(scheme_name, scheme):
    """Run setup + full training for one scheme, return result dict."""
    init_t = scheme.setup()
    result = scheme.run()
    result["init_time_s"] = init_t   # override with measured setup time
    return result


def run_task(model_key, dataset_name):
    """
    Run all schemes on one (model, dataset) task, averaged over N_REPEAT.
    Returns a dict: scheme_name → averaged metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Task: MLP-{model_key}  |  Dataset: {dataset_name}")
    print(f"{'='*60}")

    loader = LOADERS[dataset_name]
    dataset = loader()

    aggregated = {}   # scheme_name → list of result dicts

    for rep in range(N_REPEAT):
        print(f"  Repetition {rep+1}/{N_REPEAT}")
        schemes = build_schemes(model_key, dataset)

        for name, scheme in schemes:
            print(f"    Running {name} ...", end=" ", flush=True)
            t_start = time.perf_counter()
            result = run_single(name, scheme)
            elapsed = time.perf_counter() - t_start
            print(f"done ({elapsed:.1f}s)")

            if name not in aggregated:
                aggregated[name] = []
            aggregated[name].append(result)

    # Average over repetitions
    averaged = {}
    for name, results in aggregated.items():
        keys = [k for k in results[0] if k != "scheme"]
        averaged[name] = {"scheme": name}
        for k in keys:
            vals = [r[k] for r in results if isinstance(r[k], (int, float))]
            averaged[name][k] = float(np.mean(vals)) if vals else 0.0

    return averaged


def run_all_experiments():
    """Run all three tasks and save results."""
    all_results = {}

    for model_key, dataset_name in TASKS:
        task_label = f"MLP-{model_key}" if "ResNet" not in model_key else model_key
        results = run_task(model_key, dataset_name)
        all_results[task_label] = results

        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    print("\n\nAll experiments complete.")
    _print_summary(all_results)
    return all_results


def _print_summary(all_results):
    """Print a compact summary table to stdout."""
    metrics = [
        ("init_time_s",        "Init time (s)"),
        ("single_prove_time_s","Single prove (s)"),
        ("total_prove_time_s", "Total prove (s)"),
        ("verify_time_s",      "Verify time (s)"),
        ("storage_mb",         "Storage (MB)"),
    ]
    schemes = ["zkPMT", "Garg", "Kaizen", "zkCNN", "zkDL", "VeriCNN"]

    for task, results in all_results.items():
        print(f"\n── {task} ──")
        header = f"{'Metric':<25}" + "".join(f"{s:>12}" for s in schemes)
        print(header)
        print("-" * len(header))
        for key, label in metrics:
            row = f"{label:<25}"
            for s in schemes:
                val = results.get(s, {}).get(key, float("nan"))
                row += f"{val:>12.3f}"
            print(row)
