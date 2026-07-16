"""
zkPMT Experiment Entry Point
=============================
Reproduces the experimental evaluation from the paper:
  "Zero-Knowledge Proof-Based Integrity Verification for DNN Training (zkPMT)"

Usage
-----
  # Run full experiments (slow, ~hours depending on hardware):
  python main.py

  # Run with reduced repetitions for quick validation:
  python main.py --fast

  # Only plot (requires results/results.json from a previous run):
  python main.py --plot-only

Output
------
  results/results.json   – raw timing and storage numbers
  figures/fig4_*.png     – initialization cost (Fig. 4)
  figures/fig5_*.png     – single-round prove cost (Fig. 5)
  figures/fig6_*.png     – cumulative prove cost (Fig. 6)
  figures/fig7_*.png     – verification cost (Fig. 7)
  figures/fig8_*.png     – storage cost (Fig. 8)
  figures/fig6c_*.png    – scalability line chart (Fig. 6c)
"""

import sys
import argparse

import experiments
import plot_results


def parse_args():
    p = argparse.ArgumentParser(description="zkPMT experiments")
    p.add_argument("--fast",      action="store_true",
                   help="Use N_REPEAT=1 and NUM_ROUNDS=20 for quick testing")
    p.add_argument("--plot-only", action="store_true",
                   help="Only generate plots from existing results/results.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        print("=" * 60)
        print("  Mode: PLOT-ONLY  (reading results/results.json)")
        print("=" * 60)
        results = plot_results.load_results(use_mock=False)
        plot_results.plot_all(results)
        return

    if args.fast:
        print("=" * 60)
        print("  Mode: FAST  (N_REPEAT=1, NUM_ROUNDS=20)")
        print("=" * 60)
        experiments.N_REPEAT   = 1
        experiments.NUM_ROUNDS = 20

    print("=" * 60)
    print("  zkPMT Experimental Evaluation")
    print("  Schemes: zkPMT | Garg | Kaizen | zkCNN | zkDL | VeriCNN")
    print("  Tasks:   MLP-Small+Iris | MLP-Medium+MNIST | MLP-Large+CIFAR-10")
    print("  Metrics: Init | Prove | Verify | Storage")
    print("=" * 60)

    # Run all experiments
    results = experiments.run_all_experiments()

    # Generate figures
    print("\nGenerating figures ...")
    plot_results.plot_all(results)

    print("\nDone. Check results/ and figures/ directories.")


if __name__ == "__main__":
    main()
