"""
Plot experimental results to reproduce Figures 4–8 from the paper.

Usage:
    python plot_results.py                  # reads results/results.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

SCHEMES   = ["Garg", "Kaizen", "zkCNN", "zkDL", "VeriCNN", "zkPMT"]
TASKS     = ["MLP-Small+Iris", "MLP-Medium+MNIST", "MLP-Large+CIFAR-10", "ResNet50+CIFAR-10"]
TASK_LABELS = ["MLP-Small\n+Iris", "MLP-Medium\n+MNIST", "MLP-Large\n+CIFAR-10", "ResNet50\n+CIFAR-10"]

# Colour palette (one per scheme, consistent across all figures)
COLORS = {
    "Garg":    "#e74c3c",
    "Kaizen":  "#e67e22",
    "zkCNN":   "#2ecc71",
    "zkDL":    "#3498db",
    "VeriCNN": "#9b59b6",
    "zkPMT":   "#1abc9c",
}

def _empty_results_template():
    """Return an empty placeholder result structure without concrete values."""
    return {task: {scheme: {} for scheme in SCHEMES} for task in TASKS}


def load_results():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, using empty placeholders.")
        return _empty_results_template()
    with open(path) as f:
        return json.load(f)


def _bar_figure(results, metric_key, ylabel, title, filename,
                skip_schemes=None, note=None):
    """
    Generic grouped bar chart: x-axis = tasks, groups = schemes.
    """
    skip_schemes = skip_schemes or []
    active = [s for s in SCHEMES if s not in skip_schemes]

    x = np.arange(len(TASKS))
    width = 0.13
    offsets = np.linspace(-(len(active)-1)/2, (len(active)-1)/2, len(active)) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, scheme in enumerate(active):
        vals = []
        for task in TASKS:
            v = results.get(task, {}).get(scheme, {}).get(metric_key, 0.0)
            vals.append(v)
        bars = ax.bar(x + offsets[i], vals, width,
                      label=scheme, color=COLORS[scheme], alpha=0.85,
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=3)
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


def plot_all(results):
    print("\nGenerating figures ...")

    # Fig. 4 – Initialization cost
    _bar_figure(results, "init_time_s", "Time (s)",
                "Fig. 4  Initialization Cost",
                "fig4_init_cost.png")

    # Fig. 5 – Single-round proof generation cost (zkCNN excluded)
    _bar_figure(results, "single_prove_time_s", "Time (s)",
                "Fig. 5  Single-Round Proof Generation Cost",
                "fig5_single_prove.png",
                skip_schemes=["zkCNN"],
                note="zkCNN does not generate training proofs (forward-only).")

    # Fig. 6 – Cumulative proof generation cost (zkCNN excluded)
    _bar_figure(results, "total_prove_time_s", "Time (s)",
                "Fig. 6  Cumulative Proof Generation Cost (100 rounds)",
                "fig6_total_prove.png",
                skip_schemes=["zkCNN"],
                note="zkPMT aggregates every 10 rounds; advantage grows with T.")

    # Fig. 7 – Verification cost (zkCNN excluded)
    _bar_figure(results, "verify_time_s", "Time (s)",
                "Fig. 7  Verification Cost",
                "fig7_verify.png",
                skip_schemes=["zkCNN"],
                note="zkPMT: one Groth16 call per 10 rounds (recursive aggregation).")

    # Fig. 8 – Storage cost
    _bar_figure(results, "storage_mb", "Storage (MB)",
                "Fig. 8  Storage Cost",
                "fig8_storage.png")

    # Bonus: cumulative prove cost vs. number of rounds (line chart)
    _plot_scalability(results)


def _plot_scalability(results):
    """
    Line chart: cumulative prove time vs. number of rounds for
    MLP-Large+CIFAR-10, extrapolated from per-round costs.
    Reproduces the trend shown in Fig. 6(c).
    """
    task = "MLP-Large+CIFAR-10"
    task_data = results.get(task, {})
    round_counts = [10, 20, 30, 50, 70, 100]

    fig, ax = plt.subplots(figsize=(8, 5))

    for scheme in SCHEMES:
        if scheme == "zkCNN":
            continue
        single = task_data.get(scheme, {}).get("single_prove_time_s", 0.0)
        if scheme == "zkPMT":
            # zkPMT aggregates every 10 rounds: cost ≈ single_prove * T/10
            cumulative = [single * (r / 10) for r in round_counts]
        else:
            cumulative = [single * r for r in round_counts]
        ax.plot(round_counts, cumulative, marker="o", label=scheme,
                color=COLORS[scheme], linewidth=2)

    ax.set_xlabel("Number of Training Rounds", fontsize=11)
    ax.set_ylabel("Cumulative Prove Time (s)", fontsize=11)
    ax.set_title("Fig. 6(c) Scalability – MLP-Large + CIFAR-10", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig6c_scalability.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    results = load_results()
    plot_all(results)
    print("\nAll figures saved to", FIGURES_DIR)
