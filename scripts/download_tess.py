"""Download TESS two-minute SPOC light curves for the labelled TOI cross-mission set.

MAST publishes, per sector, a shell script listing every light-curve product in
that sector.  Parsing those listings gives a complete TIC -> file index without
issuing thousands of archive queries; the files themselves are then pulled from
the public AWS mirror, whose path layout is derivable from the file name.

Usage::

    python scripts/download_tess.py --max-sectors 4 --threads 96
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

SCRIPT_INDEX = "https://archive.stsci.edu/missions/tess/download_scripts/sector/"
S3_ROOT = "https://stpubdata.s3.amazonaws.com/tess/public/tid"
FILE_RE = re.compile(r"(tess\d+-s(\d{4})-(\d{16})-\d+-s_lc\.fits)")
MIN_FITS_BYTES = 50_000

# TFOPWG dispositions that carry a trustworthy label.
POSITIVE = {"CP", "KP"}   # confirmed / known planet
NEGATIVE = {"FP", "FA"}   # false positive / false alarm


def s3_url(fname: str, tic16: str) -> str:
    """AWS path for a light-curve file: the TIC is split into four 4-digit groups."""
    sector = fname.split("-")[1]  # e.g. s0001
    parts = [tic16[i : i + 4] for i in range(0, 16, 4)]
    return f"{S3_ROOT}/{sector}/{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}/{fname}"


def fetch_bytes(url: str, timeout: int = 180) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def list_sector_scripts() -> list[str]:
    html = fetch_bytes(SCRIPT_INDEX)
    if html is None:
        raise RuntimeError("could not reach the MAST sector script index")
    names = sorted(set(re.findall(r"tesscurl_sector_(\d+)_lc\.sh", html.decode())), key=int)
    return [f"{SCRIPT_INDEX}tesscurl_sector_{n}_lc.sh" for n in names]


def build_index(threads: int) -> dict[str, list[tuple[int, str]]]:
    """Map zero-padded TIC -> [(sector, filename), ...] across all sectors."""
    urls = list_sector_scripts()
    print(f"parsing {len(urls)} sector listings", flush=True)
    index: dict[str, list[tuple[int, str]]] = {}
    with cf.ThreadPoolExecutor(min(threads, 24)) as pool:
        for blob in pool.map(fetch_bytes, urls):
            if blob is None:
                continue
            for fname, sector, tic in FILE_RE.findall(blob.decode(errors="ignore")):
                index.setdefault(tic, []).append((int(sector), fname))
    for tic in index:
        index[tic].sort()
    print(f"indexed {len(index):,} TIC targets", flush=True)
    return index


def fetch(url: str, dest: str, retries: int = 3) -> int:
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_FITS_BYTES:
        return os.path.getsize(dest)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                blob = r.read()
            if len(blob) < MIN_FITS_BYTES:
                return 0
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, dest)
            return len(blob)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 0
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toi-table", default="data/catalogs/toi.csv")
    ap.add_argument("--out-dir", default="data/tess_lc")
    ap.add_argument("--index-out", default="data/catalogs/tess_file_index.csv")
    ap.add_argument("--max-sectors", type=int, default=4)
    ap.add_argument("--threads", type=int, default=96)
    args = ap.parse_args()

    toi = pd.read_csv(args.toi_table, low_memory=False)
    keep = toi["tfopwg_disp"].isin(POSITIVE | NEGATIVE)
    keep &= toi[["pl_tranmid", "pl_orbper", "pl_trandurh"]].notna().all(axis=1)
    toi = toi[keep]
    tics = sorted({f"{int(t):016d}" for t in toi["tid"].unique()})
    print(f"{len(toi):,} labelled TOIs over {len(tics):,} TIC targets", flush=True)

    index = build_index(args.threads)

    jobs, rows = [], []
    for tic in tics:
        for sector, fname in index.get(tic, [])[: args.max_sectors]:
            dest = os.path.join(args.out_dir, tic, fname)
            jobs.append((s3_url(fname, tic), dest))
            rows.append({"tic": int(tic), "sector": sector, "filename": fname})
    pd.DataFrame(rows).to_csv(args.index_out, index=False)
    covered = len({r["tic"] for r in rows})
    print(
        f"{len(jobs):,} files for {covered:,}/{len(tics):,} targets "
        f"(<= {args.max_sectors} sectors each)",
        flush=True,
    )

    done = total = 0
    start = time.time()
    with cf.ThreadPoolExecutor(args.threads) as pool:
        futures = [pool.submit(fetch, u, d) for u, d in jobs]
        for fut in cf.as_completed(futures):
            total += fut.result()
            done += 1
            if done % 1000 == 0:
                el = time.time() - start
                print(
                    f"[{done:,}/{len(jobs):,}] {total/1e9:.1f} GB {total/el/1e6:.1f} MB/s "
                    f"ETA {(len(jobs)-done)/(done/el)/60:.0f} min",
                    flush=True,
                )
    el = time.time() - start
    print(f"done: {total/1e9:.1f} GB in {el/60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
