import argparse
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = (
    "T1_T6_ct_reconstruction",
    "T1_ct_denoising",
    "T3_ct_segmentation",
    "T2_mri_reconstruction",
    "T4_mri_segmentation",
    "T5_pathology_classification",
    "T7_privacy_protection",
)


def main():
    parser = argparse.ArgumentParser(description="CUDIRCL experiments")
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Task command and options; use EXPERIMENT --help")
    args = parser.parse_args()
    script = Path(__file__).resolve().with_name(f"{args.experiment}.py")
    return subprocess.run([sys.executable, str(script), *args.arguments]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
