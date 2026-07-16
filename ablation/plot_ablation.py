"""
Ablation Study Plots.

Generates four figures:
  ablation_fig1_init.png        – initialization cost per variant
  ablation_fig2_prove.png       – total proof generation cost per variant
  ablation_fig3_verify.png      – verification cost per variant
  ablation_fig4_storage.png     – storage cost per variant
  ablation_fig5_gates.png       – circuit gate count per variant
  ablation_fig6_contribution.png– normalized contribution of each component
                                  (relative increase when removed)

Usage:
    python plot_ablation.py              # reads results/ablation_results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ablation_variants import ALL_VARIANTS, VARIANT_COLORS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TASKS       = ["MLP-Small+Iris", "MLP-Medium+MNIST", "MLP-Large+CIFAR-10", "ResNet50+CIFAR-10"]
TASK_LABELS = ["MLP-Small\n+Iris", "MLP-Medium\n+MNIST", "MLP-Large\n+CIFAR-10", "ResNet50\n+CIFAR-10"]
VARIANTS    = [v.name for v in ALL_VARIANTS]


def _empty_results_template():
    """Return an empty placeholder result structure without concrete values."""
    return {task: {variant: {} for variant in VARIANTS} for task in TASKS}


def load_results():
    path = os.path.join(RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, using empty placeholders.")
        return _empty_results_template()
    with open(path) as f:
        return json.load(f)


def _grouped_bar(results, metric_key, ylabel, title, filename, note=None):
    """Grouped bar chart: x = tasks, groups = variants."""
    x = np.arange(len(TASKS))
    n = len(VARIANTS)
    width = 0.13
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, variant in enumerate(VARIANTS):
        vals = [results.get(t, {}).get(variant, {}).get(metric_key, 0.0)
                for t in TASKS]
        ax.bar(x + offsets[i], vals, width,
               label=variant, color=VARIANT_COLORS[variant],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    if note:
        ax.text(0.01, 0.98, note, transform=ax.transAxes,
                fontsize=7, va="top", color="gray")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_contribution(results):
    """
    Fig. 6: Normalized contribution of each component.
    For each ablated variant, compute the relative overhead increase
    compared to zkPMT-Full across all tasks and metrics.

    contribution(C) = mean over tasks of
        (metric(w/o-C) - metric(Full)) / metric(Full)
    """
    ablated = [v for v in VARIANTS if v != "zkPMT-Full"]
    metrics = ["init_time_s", "total_prove_time_s", "verify_time_s", "storage_mb"]
    metric_labels = ["Init", "Total Prove", "Verify", "Storage"]

    # contribution[variant][metric] = mean relative increase across tasks
    contrib = {v: [] for v in ablated}
    for v in ablated:
        for m in metrics:
            diffs = []
            for t in TASKS:
                full_val = results.get(t, {}).get("zkPMT-Full", {}).get(m, 1.0)
                ablated_val = results.get(t, {}).get(v, {}).get(m, 1.0)
                if full_val > 0:
                    diffs.append((ablated_val - full_val) / full_val)
            contrib[v].append(float(np.mean(diffs)) if diffs else 0.0)

    x = np.arange(len(metric_labels))
    n = len(ablated)
    width = 0.14
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(ablated):
        ax.bar(x + offsets[i], contrib[v], width,
               label=v, color=VARIANT_COLORS[v],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Relative Overhead Increase vs. zkPMT-Full", fontsize=10)
    ax.set_title("Ablation Fig. 6  Component Contribution\n"
                 "(higher bar = removing this component hurts more)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ablation_fig6_contribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_all(results):
    print("\nGenerating ablation figures ...")

    _grouped_bar(results, "init_time_s", "Time (s)",
                 "Ablation Fig. 1  Initialization Cost",
                 "ablation_fig1_init.png")

    _grouped_bar(results, "total_prove_time_s", "Time (s)",
                 "Ablation Fig. 2  Total Proof Generation Cost",
                 "ablation_fig2_prove.png")

    _grouped_bar(results, "verify_time_s", "Time (s)",
                 "Ablation Fig. 3  Verification Cost",
                 "ablation_fig3_verify.png",
                 note="w/o-Recursion verifies every sub-proof individually → O(T) cost.")

    _grouped_bar(results, "storage_mb", "Storage (MB)",
                 "Ablation Fig. 4  Storage Cost",
                 "ablation_fig4_storage.png")

    _grouped_bar(results, "num_gates", "Gate Count",
                 "Ablation Fig. 5  Circuit Gate Count",
                 "ablation_fig5_gates.png",
                 note="w/o-FixedPoint: ~2.5× more gates due to multi-scale scaling/rounding.")

    plot_contribution(results)


if __name__ == "__main__":
    results = load_results()
    plot_all(results)
    print("\nAll ablation figures saved to", FIGURES_DIR)
