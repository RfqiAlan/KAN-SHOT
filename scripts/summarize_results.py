"""
Master Result Summarizer for KAN-SHOT Manuscript (Q1 Target).
Parses evaluation CSVs and ablation logs to print:
  - Table 4: Near-domain CRC-VAL (9-way 5-shot)
  - Table 5: Cross-domain LC 5-way (1-shot, 5-shot, 10-shot)
  - Table 6: Domain Bias (LC-Colon vs LC-Lung)
  - Table 7: Episodic Fine-tuning Ablations (1,500 episodes)
  - Table 8: Image Corruption Robustness (1,500 episodes)
"""
import os
import csv
import glob
import re
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def load_stat_from_csv(csv_path):
    """Load MEAN and CI95 from an aggregated CSV."""
    if not os.path.exists(csv_path):
        return None, None
    mean_val, ci_val = None, None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) >= 2:
                if r[0] == "MEAN":
                    mean_val = float(r[1])
                elif r[0] == "CI95":
                    ci_val = float(r[1])
    return mean_val, ci_val


def print_table_4():
    print("\n" + "=" * 65)
    print("TABLE 4: Near-Domain Results (CRC-VAL 9-way 5-shot, 3,000 episodes)")
    print("=" * 65)
    print(f"{'Method':<25} {'Accuracy (%)':<18} {'95% CI':<15}")
    print("-" * 65)

    methods = [
        ("ProtoNet (frozen)", "protonet_crc_val_shot5_aggregated.csv"),
        ("KAN-SHOT Distance (frozen)", "kanprotonet_crc_val_shot5_aggregated.csv"),
        ("MLP-ProtoNet (frozen)", "mlpprotonet_crc_val_shot5_aggregated.csv"),
    ]
    for label, fname in methods:
        path = os.path.join(RESULTS_DIR, "multiseed", fname)
        mean, ci = load_stat_from_csv(path)
        if mean is not None:
            print(f"{label:<25} {mean:<18.2f} ±{ci:.2f}")
        else:
            print(f"{label:<25} {'[Pending evaluation]':<18}")
    print("=" * 65)


def print_table_5():
    print("\n" + "=" * 65)
    print("TABLE 5: Cross-Domain Results: LC 5-way (Frozen Backbone, 3,000 ep)")
    print("=" * 65)
    print(f"{'Method':<20} {'1-Shot':<15} {'5-Shot':<15} {'10-Shot':<15}")
    print("-" * 65)

    configs = [
        ("ProtoNet", "protonet"),
        ("KAN-SHOT", "kanprotonet"),
        ("MLP-ProtoNet", "mlpprotonet"),
    ]
    for label, prefix in configs:
        shots_res = []
        for s in [1, 5, 10]:
            fname = f"{prefix}_lc5way_shot{s}_aggregated.csv"
            path = os.path.join(RESULTS_DIR, "multiseed", fname)
            m, _ = load_stat_from_csv(path)
            shots_res.append(f"{m:.2f}%" if m is not None else "N/A")
        print(f"{label:<20} {shots_res[0]:<15} {shots_res[1]:<15} {shots_res[2]:<15}")
    print("=" * 65)


def print_table_6():
    print("\n" + "=" * 65)
    print("TABLE 6: Domain Bias: LC-Colon vs LC-Lung (5-way 5-shot, 3,000 ep)")
    print("=" * 65)
    print(f"{'Method':<20} {'LC-Colon':<20} {'LC-Lung':<20}")
    print("-" * 65)

    methods = [
        ("ProtoNet", "protonet"),
        ("KAN-SHOT", "kanprotonet"),
    ]
    for label, prefix in methods:
        colon_f = f"{prefix}_lc_colon_shot5_aggregated.csv"
        lung_f = f"{prefix}_lc_lung_shot5_aggregated.csv"
        m_colon, _ = load_stat_from_csv(os.path.join(RESULTS_DIR, "multiseed", colon_f))
        m_lung, _ = load_stat_from_csv(os.path.join(RESULTS_DIR, "multiseed", lung_f))
        c_str = f"{m_colon:.2f}%" if m_colon else "N/A"
        l_str = f"{m_lung:.2f}%" if m_lung else "N/A"
        print(f"{label:<20} {c_str:<20} {l_str:<20}")
    print("=" * 65)


def print_table_7():
    print("\n" + "=" * 65)
    print("TABLE 7: Episodic Fine-Tuning Ablation: LC 5-way 5-shot (1,500 ep)")
    print("=" * 65)
    print(f"{'Strategy':<22} {'Unfrozen':<25} {'Acc (%)':<10} {'95% CI':<8}")
    print("-" * 65)

    strategies = [
        ("Frozen", "None", None, 58.37, 0.48),
        ("FT-KAN", "KAN only", "kan_lc5way_5shot_finetune_kan_seed*.csv", None, None),
        ("FT-Layer4", "Layer4 + KAN", "kan_lc5way_5shot_finetune_layer4_seed*.csv", None, None),
        ("FT-ALL", "All + KAN", "kan_lc5way_5shot_finetune_all_seed*.csv", None, None),
        ("FT-ALL (freeze phi)", "All + KAN (spline frozen)", "kan_lc5way_5shot_finetune_all_freeze_spline_seed*.csv", None, None),
        ("FT-ALL (ProtoNet)", "All backbone", "protonet_lc5way_5shot_finetune_all_seed*.csv", None, None),
    ]

    ft_dir = os.path.join(RESULTS_DIR, "FT_results")
    for name, unfrozen, pat, fixed_m, fixed_ci in strategies:
        if fixed_m is not None:
            print(f"{name:<22} {unfrozen:<25} {fixed_m:<10.2f} ±{fixed_ci:.2f}")
            continue

        files = sorted(glob.glob(os.path.join(ft_dir, pat)))
        accs = []
        for f in files:
            with open(f, 'r') as fp:
                for row in csv.reader(fp):
                    if len(row) == 2 and row[0].isdigit():
                        accs.append(float(row[1]))
        if accs:
            arr = np.array(accs) * 100
            mean = np.mean(arr)
            ci95 = 1.96 * (np.std(arr) / np.sqrt(len(arr)))
            print(f"{name:<22} {unfrozen:<25} {mean:<10.2f} ±{ci95:.2f}")
        else:
            print(f"{name:<22} {unfrozen:<25} {'[Pending]':<10}")
    print("=" * 65)


def print_table_8():
    print("\n" + "=" * 65)
    print("TABLE 8: Robustness Under Image Corruption (LC 5-way 5-shot, 1,500 ep)")
    print("=" * 65)
    print(f"{'Corruption':<22} {'Severity 1':<14} {'Severity 2':<14} {'Severity 3':<14}")
    print("-" * 65)

    rob_dir = os.path.join(RESULTS_DIR, "robustness")
    corruptions = {}
    for seed in [2021, 2022, 2023]:
        fpath = os.path.join(rob_dir, f"robustness_seed{seed}.txt")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-16") as f:
                text = f.read()
        except Exception:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        matches = re.finditer(r"--- (.*?) ---\n\s*Severity 1: (.*?)%\n\s*Severity 2: (.*?)%\n\s*Severity 3: (.*?)%", text)
        for match in matches:
            c, s1, s2, s3 = match.groups()
            if c not in corruptions:
                corruptions[c] = [[], [], []]
            corruptions[c][0].append(float(s1))
            corruptions[c][1].append(float(s2))
            corruptions[c][2].append(float(s3))

    for c in ["gaussian_blur", "gaussian_noise", "stain_jitter", "jpeg_compression", "brightness_shift"]:
        if c in corruptions:
            s1, s2, s3 = corruptions[c]
            c_name = c.replace("_", " ").title()
            print(f"{c_name:<22} {np.mean(s1):<14.2f} {np.mean(s2):<14.2f} {np.mean(s3):<14.2f}")
    print("-" * 65)
    print("Clean Baseline: 58.37%")
    print("=" * 65)


def main():
    print("\n" + "#" * 65)
    print("# KAN-SHOT MANUSCRIPT EMPIRICAL RESULTS SUMMARY (TABLES 4 - 8)")
    print("#" * 65)
    print_table_4()
    print_table_5()
    print_table_6()
    print_table_7()
    print_table_8()


if __name__ == "__main__":
    main()

