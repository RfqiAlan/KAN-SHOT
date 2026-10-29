"""
Statistical Significance Analysis for KAN-SHOT vs ProtoNet Baselines.
Performs Wilcoxon signed-rank tests with Bonferroni correction and Cohen's d
across paired few-shot episodes (1,000 to 3,000 episodes).
"""
import os
import sys
import csv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kan_shot.evaluation.statistical_tests import pairwise_comparison, format_report

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "multiseed")


def load_accuracies(csv_path):
    """Load per-episode accuracy floats from a CSV file."""
    accs = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) == 2 and r[0].isdigit():
                accs.append(float(r[1]))
    return np.array(accs)


def main():
    print("=" * 65)
    print("STATISTICAL ROBUSTNESS TESTS: Wilcoxon Signed-Rank + Cohen's d")
    print("=" * 65)

    proto_csv = os.path.join(RESULTS_DIR, "protonet_lc5way_shot5_aggregated.csv")
    kan_csv = os.path.join(RESULTS_DIR, "kanprotonet_lc5way_shot5_aggregated.csv")
    mlp_csv = os.path.join(RESULTS_DIR, "mlpprotonet_lc5way_shot5_aggregated.csv")

    if not os.path.exists(proto_csv) or not os.path.exists(kan_csv):
        print("⚠️ Aggregated CSV files not found in results/multiseed/")
        return

    proto_accs = load_accuracies(proto_csv)
    kan_accs = load_accuracies(kan_csv)

    models = {
        "ProtoNet": proto_accs,
        "KAN-Distance": kan_accs
    }

    if os.path.exists(mlp_csv):
        models["MLP-ProtoNet"] = load_accuracies(mlp_csv)

    report = pairwise_comparison(models, "LC 5-way 5-shot (Frozen Backbone, 3,000 episodes)")
    formatted = format_report(report)
    print(formatted)

    output_path = os.path.join(PROJECT_ROOT, "results", "wilcoxon_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted)
    print(f"\n💾 Saved statistical report to: {output_path}")


if __name__ == "__main__":
    main()

