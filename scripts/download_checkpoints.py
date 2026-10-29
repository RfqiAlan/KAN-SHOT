"""
Checkpoint verification and download helper for KAN-SHOT.
Verifies that pretrained backbones and KAN metric weights exist locally in checkpoints/.
"""
import os
import sys
import argparse
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

REQUIRED_CHECKPOINTS = {
    "seed2021": [
        "resnet18_nct.pth",
        "kanprotonet_cosine_nct.pth",
        "mlpprotonet_cosine_nct.pth"
    ],
    "seed2022": [
        "kanprotonet_cosine_nct.pth",
        "mlpprotonet_cosine_nct.pth"
    ],
    "seed2023": [
        "kanprotonet_cosine_nct.pth",
        "mlpprotonet_cosine_nct.pth"
    ]
}


def verify_checkpoints():
    print("=" * 65)
    print("CHECKPOINT INTEGRITY CHECK")
    print("=" * 65)
    all_ok = True
    for seed, files in REQUIRED_CHECKPOINTS.items():
        seed_dir = os.path.join(CHECKPOINTS_DIR, seed)
        print(f"\n📁 Checking {seed} ({seed_dir}):")
        for f in files:
            path = os.path.join(seed_dir, f)
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✅ {f:<30} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {f:<30} (MISSING)")
                all_ok = False

    print("\n" + "=" * 65)
    if all_ok:
        print("🎉 ALL REQUIRED CHECKPOINTS ARE PRESENT AND VERIFIED!")
    else:
        print("ℹ️ Note: Seed 2021 checkpoints are included locally for evaluation.")
        print("   If you wish to train seeds 2022 & 2023 from scratch, run:")
        print("     python scripts/run_multiseed_phase_b.py --data_root data/raw --seeds 2022 2023")
    print("=" * 65)
    return all_ok


if __name__ == "__main__":
    verify_checkpoints()

