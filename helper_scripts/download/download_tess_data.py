"""
Download TESS light curves for one or more TIC IDs via Lightkurve/MAST.

Storage-conscious defaults:
  - Downloads only 1 sector per target (latest available)
  - Single-aperture SPOC product (smallest file size)
  - Saves as compressed CSV (~100-200 KB per target)
  - Use --sectors N to download more sectors if needed

Usage:
    python helper_scripts/download/download_tess_data.py TIC 307210830 TIC 25155310
    python helper_scripts/download/download_tess_data.py --file target_list.txt --sectors 2
    python helper_scripts/download/download_tess_data.py --list-known   # small benchmark set
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# Small benchmark set of confirmed planet hosts — ~1 sector each ≈ 200 MB total
KNOWN_HOSTS = [
    "TIC 307210830",   # Pi Mensae b  — hot Jupiter, single sector
    "TIC 25155310",    # TOI-270      — multi-planet system
    "TIC 150428135",   # LHS 3844 b   — ultra-short period rocky planet
    "TIC 100100827",   # HD 219134    — multi-planet, bright host
    "TIC 264985538",   # TOI-700 d    — habitable-zone Earth-sized planet
]


def resolve_tic_ids(args) -> list[str]:
    ids: list[str] = []
    if args.list_known:
        ids.extend(KNOWN_HOSTS)
    if args.file:
        with open(args.file) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    ids.append(stripped)
    if args.tic_ids:
        ids.extend(args.tic_ids)
    if not ids:
        print("No TIC IDs provided. Use --list-known, --file, or pass IDs directly.")
        sys.exit(1)
    return ids


def download_target(tic_id: str, sectors: int | None, data_dir: Path) -> dict:
    import lightkurve as lk

    print(f"  Searching for {tic_id} ...", end=" ")
    search = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC")
    if len(search) == 0:
        print("no SPOC data found — skipping")
        return {"tic_id": tic_id, "status": "not_found", "files": []}

    n_available = len(search)
    n_dl = sectors if sectors is not None else 1
    n_dl = min(n_dl, n_available)

    files = []
    for i in range(n_dl):
        row = search[i]
        sector_num = row.description.split("Sector ")[-1].split(" ")[0] if "Sector" in row.description else f"idx{i}"
        print(f"Sector {sector_num}", end=" ")
        lc = row.download()
        lc = lc.remove_nans().remove_outliers(sigma=5)

        safe_name = tic_id.replace(" ", "_")
        csv_path = data_dir / f"{safe_name}_sector{sector_num}.csv"
        df = pd.DataFrame({
            "time": lc.time.value,
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value,
        })
        df.to_csv(csv_path, index=False)
        files.append(str(csv_path))
        print(f"→ {csv_path.name} ({lc.time.value.size} pts)", end=" ")

    print()
    return {"tic_id": tic_id, "status": "downloaded", "files": files, "sectors_available": n_available}


def main():
    parser = argparse.ArgumentParser(description="Download TESS light curves (storage-conscious defaults)")
    parser.add_argument("tic_ids", nargs="*", help="One or more TIC IDs, e.g. TIC 307210830")
    parser.add_argument("--file", type=str, default=None, help="File with one TIC ID per line")
    parser.add_argument("--list-known", action="store_true", help="Download small benchmark set of known planet hosts")
    parser.add_argument("--sectors", type=int, default=None,
                        help="Sectors per target (default: 1 — keeps download small)")
    parser.add_argument("--data-dir", type=str, default="data/tess",
                        help="Output directory (default: data/tess)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tic_ids = resolve_tic_ids(args)
    print(f"Downloading {len(tic_ids)} target(s) to {data_dir} ...\n")

    manifest = []
    for tic_id in tic_ids:
        result = download_target(tic_id, sectors=args.sectors, data_dir=data_dir)
        manifest.append(result)
        print()

    manifest_path = data_dir / "download_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"Manifest saved to {manifest_path}")

    n_ok = sum(1 for r in manifest if r["status"] == "downloaded")
    print(f"Done: {n_ok}/{len(tic_ids)} targets downloaded successfully.")


if __name__ == "__main__":
    main()
