"""
Download the curated labeled dataset for training the classification model.

The curated dataset (provided by ISRO during the hackathon induction) contains
labeled light curves across categories:
  - confirmed_exoplanet
  - eclipsing_binary
  - false_positive (blend / starspot / noise)
  - unknown

This script downloads the dataset from a configurable source URL or local path,
validates its structure, and organizes it into `data/curated/` in a format that
the pipeline's training code (Mamba classifier, Phase 2) can consume directly.

Usage:
    python helper_scripts/download/download_curated_dataset.py
    python helper_scripts/download/download_curated_dataset.py --url <download_url>
    python helper_scripts/download/download_curated_dataset.py --local-path /path/to/dataset.zip
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd


CATEGORIES = ["confirmed_exoplanet", "eclipsing_binary", "false_positive", "unknown"]
REQUIRED_COLUMNS = ["tic_id", "label", "time", "flux", "flux_err"]


def validate_structure(data_dir: Path) -> dict:
    report = {"valid": True, "categories": {}, "total_curves": 0, "issues": []}
    for cat in CATEGORIES:
        cat_dir = data_dir / cat
        if not cat_dir.exists():
            report["issues"].append(f"Missing category directory: {cat}/")
            report["valid"] = False
            continue
        csv_files = sorted(cat_dir.glob("*.csv"))
        report["categories"][cat] = {
            "present": cat_dir.exists(),
            "n_curves": len(csv_files),
        }
        report["total_curves"] += len(csv_files)
        for f in csv_files:
            try:
                df = pd.read_csv(f, nrows=1)
                missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    report["issues"].append(f"{f.name}: missing columns {missing}")
                    report["valid"] = False
            except Exception as e:
                report["issues"].append(f"{f.name}: read error — {e}")
                report["valid"] = False
    return report


def download_from_url(url: str, dest: Path) -> Path:
    import urllib.request

    archive_path = dest / "curated_dataset.zip"
    print(f"Downloading curated dataset from {url} ...")
    urllib.request.urlretrieve(url, archive_path)
    print(f"Downloaded ({archive_path.stat().st_size / 1e6:.1f} MB)")
    return archive_path


def extract_archive(archive_path: Path, dest: Path):
    print(f"Extracting {archive_path.name} to {dest} ...")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)
    print("Extraction complete.")


def print_report(report: dict):
    print(f"\nDataset validation {'✅ PASSED' if report['valid'] else '❌ HAS ISSUES'}")
    print(f"Total light curves: {report['total_curves']}")
    for cat, info in report["categories"].items():
        print(f"  {cat}: {info['n_curves']} curves")
    if report["issues"]:
        print("\nIssues:")
        for issue in report["issues"]:
            print(f"  • {issue}")


def main():
    parser = argparse.ArgumentParser(description="Download curated labeled dataset for training")
    parser.add_argument("--url", type=str, default=None,
                        help="Download URL (if dataset is hosted remotely)")
    parser.add_argument("--local-path", type=str, default=None,
                        help="Local path to dataset archive (if already downloaded)")
    parser.add_argument("--data-dir", type=str, default="data/curated",
                        help="Output directory (default: data/curated)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing dataset structure, don't download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.validate_only:
        if not data_dir.exists():
            print(f"Dataset directory {data_dir} does not exist.")
            return
        report = validate_structure(data_dir)
        print_report(report)
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    if args.local_path:
        archive_path = Path(args.local_path)
        if not archive_path.exists():
            print(f"Local archive not found: {archive_path}")
            return
    elif args.url:
        archive_path = download_from_url(args.url, data_dir)
    else:
        print("No dataset available yet.")
        print("  The curated dataset will be provided during the hackathon induction session.")
        print(f"  Once available, place it in {data_dir} with the structure:")
        for cat in CATEGORIES:
            print(f"    {data_dir}/{cat}/  (one CSV per light curve)")
        print("\n  Or re-run with --url <download_link> or --local-path <archive.zip>")
        return

    if archive_path.suffix in (".zip",):
        extract_archive(archive_path, data_dir)

    report = validate_structure(data_dir)
    print_report(report)


if __name__ == "__main__":
    main()
