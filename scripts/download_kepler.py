"""Bulk-download Kepler long-cadence light curves for the DR24 TCE training set.

Files are pulled from the public MAST holdings on AWS S3 (``stpubdata``), which is
open access (no requester-pays) and serves the identical FITS products as the
STScI archive.  The download is resumable: a file that already exists on disk with
a plausible size is skipped, so the script can be re-run after an interruption.

Layout written to disk mirrors the archive convention::

    data/kepler/<kic[:4]>/<kic:09d>/kplr<kic:09d>-<quarter_stamp>_llc.fits

Usage::

    python scripts/download_kepler.py --threads 96
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pandas as pd

# Long-cadence quarter timestamps (Q0-Q17).  Every Kepler light-curve filename
# embeds the timestamp of the data-collection start, so the full set of URLs for
# a target can be constructed without querying the archive for a file listing.
LC_QUARTER_STAMPS = [
    "2009131105131",  # Q0
    "2009166043257",  # Q1
    "2009259160929",  # Q2
    "2009350155506",  # Q3
    "2010078095331",  # Q4
    "2010174085026",  # Q5
    "2010265121752",  # Q6
    "2010355172524",  # Q7
    "2011073133259",  # Q8
    "2011177032512",  # Q9
    "2011271113734",  # Q10
    "2012004120508",  # Q11
    "2012088054726",  # Q12
    "2012179063303",  # Q13
    "2012277125453",  # Q14
    "2013011073258",  # Q15
    "2013098041711",  # Q16
    "2013131215648",  # Q17
]

S3_ROOT = "https://stpubdata.s3.amazonaws.com/kepler/public/lightcurves"
MIN_FITS_BYTES = 20_000  # anything smaller is a truncated/failed transfer

_print_lock = threading.Lock()


def target_urls(kepid: int) -> list[tuple[str, str]]:
    """Return ``(url, relative_path)`` for every long-cadence quarter of a target."""
    kic = f"{int(kepid):09d}"
    out = []
    for stamp in LC_QUARTER_STAMPS:
        name = f"kplr{kic}-{stamp}_llc.fits"
        out.append((f"{S3_ROOT}/{kic[:4]}/{kic}/{name}", f"{kic[:4]}/{kic}/{name}"))
    return out


def fetch(url: str, dest: str, retries: int = 4) -> int:
    """Download ``url`` to ``dest``.  Returns bytes written (0 if absent/failed).

    A 404 is an expected, non-exceptional outcome: not every target was observed
    in every quarter.
    """
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_FITS_BYTES:
        return os.path.getsize(dest)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                blob = resp.read()
            if len(blob) < MIN_FITS_BYTES:
                return 0
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, dest)
            return len(blob)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # target not observed this quarter
                return 0
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tce-table", default="data/catalogs/dr24_tce.csv")
    ap.add_argument("--out-dir", default="data/kepler")
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument(
        "--labeled-only",
        action="store_true",
        default=True,
        help="restrict to TCEs carrying an av_training_set label (PC/AFP/NTP)",
    )
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of targets")
    args = ap.parse_args()

    tces = pd.read_csv(args.tce_table, low_memory=False)
    if args.labeled_only:
        tces = tces[tces["av_training_set"].isin(["PC", "AFP", "NTP"])]
    kepids = sorted(tces["kepid"].unique())
    if args.limit:
        kepids = kepids[: args.limit]

    jobs = [
        (url, os.path.join(args.out_dir, rel))
        for kepid in kepids
        for url, rel in target_urls(kepid)
    ]
    print(
        f"{len(tces):,} TCEs over {len(kepids):,} targets -> {len(jobs):,} candidate files",
        flush=True,
    )

    done = 0
    total_bytes = 0
    start = time.time()
    with cf.ThreadPoolExecutor(args.threads) as pool:
        futures = {pool.submit(fetch, url, dest): dest for url, dest in jobs}
        for fut in cf.as_completed(futures):
            total_bytes += fut.result()
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - start
                rate = total_bytes / elapsed / 1e6
                eta = (len(jobs) - done) / (done / elapsed) / 3600
                with _print_lock:
                    print(
                        f"[{done:,}/{len(jobs):,}] {total_bytes/1e9:.1f} GB "
                        f"{rate:.1f} MB/s  ETA {eta:.1f} h",
                        flush=True,
                    )

    elapsed = time.time() - start
    print(
        f"done: {total_bytes/1e9:.1f} GB in {elapsed/3600:.2f} h "
        f"({total_bytes/elapsed/1e6:.1f} MB/s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
