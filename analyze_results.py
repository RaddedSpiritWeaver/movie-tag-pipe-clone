import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# metric suffix -> display label; order defines plot/table column order.
# values are looked up with a "test_" prefix first, then "val_" for old runs.
METRICS = [
    ("f1_micro", "F1"),
    ("precision_micro", "Precision"),
    ("recall_micro", "Recall"),
    ("hamming_acc", "Hamming Acc"),
]


def get_metric(run, suffix):
    for prefix in ("test_", "val_"):
        if prefix + suffix in run:
            return run[prefix + suffix]
    return None


def load_results(results_dir):
    runs = {}
    for path in sorted(glob(os.path.join(results_dir, "*", "final_metrics.json"))):
        method = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            runs[method] = json.load(f)
    return runs


def print_table(runs):
    methods = list(runs.keys())
    name_w = max(len("method"), *(len(m) for m in methods))

    header = "method".ljust(name_w) + "".join(f"{lbl:>14}" for _, lbl in METRICS)
    print(header)
    print("-" * len(header))

    # rank by F1 (descending) for readability
    for method in sorted(methods, key=lambda m: get_metric(runs[m], "f1_micro") or 0, reverse=True):
        row = method.ljust(name_w)
        for suffix, _ in METRICS:
            val = get_metric(runs[method], suffix)
            row += f"{val:>14.4f}" if val is not None else f"{'-':>14}"
        print(row)


def plot_results(runs, out_path):
    methods = sorted(runs.keys(), key=lambda m: get_metric(runs[m], "f1_micro") or 0, reverse=True)
    n_metrics = len(METRICS)
    n_methods = len(methods)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_metrics + n_methods), 6))

    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * width
        vals = [get_metric(runs[method], suffix) or 0 for suffix, _ in METRICS]
        bars = ax.bar(x + offset, vals, width, label=method)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation metrics by method (best model)")
    ax.legend(title="method", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare final_metrics.json across methods")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <results-dir>/comparison.png)")
    args = parser.parse_args()

    runs = load_results(args.results_dir)
    if not runs:
        print(f"No final_metrics.json found under {args.results_dir}/*/")
        return

    print(f"Found {len(runs)} runs in {args.results_dir}/\n")
    print_table(runs)

    out_path = args.out or os.path.join(args.results_dir, "comparison.png")
    plot_results(runs, out_path)


if __name__ == "__main__":
    main()
