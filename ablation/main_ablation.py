"""
Ablation Study Entry Point.

Usage:
    python main_ablation.py              # full run
    python main_ablation.py --fast       # quick check (1 repeat, 20 rounds)
    python main_ablation.py --plot-only  # only plot from existing results
"""

import sys
import argparse
import run_ablation
import plot_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",      action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  zkPMT Ablation Study")
    print("  Components: FixedPoint | CircuitReuse | HashChain | Recursion")
    print("  Variants:   Full | w/o-FP | w/o-CR | w/o-HC | w/o-Rec | w/o-FP+CR")
    print("=" * 60)

    if args.plot_only:
        results = plot_ablation.load_results()
        plot_ablation.plot_all(results)
        return

    # Run experiments
    sys.argv = [sys.argv[0]] + (["--fast"] if args.fast else [])
    results = run_ablation.main()

    # Plot
    results = plot_ablation.load_results()
    plot_ablation.plot_all(results)

    print("\nDone. Check ablation/results/ and ablation/figures/")


if __name__ == "__main__":
    main()
