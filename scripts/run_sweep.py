"""Run the full experiment grid, distributing jobs across the available GPUs.

The grid covers three things the paper needs: headline comparisons over several
seeds, a component-by-component ablation, and a second split protocol for
comparability with previously published numbers.

Usage::

    python scripts/run_sweep.py --gpus 0,1
    python scripts/run_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADLINE_SEEDS = [0, 1, 2, 3, 4]
ABLATION_SEEDS = [0, 1, 2]
TCE_SPLIT_SEEDS = [0, 1, 2]

# tag -> extra command line arguments
ABLATIONS = {
    # remove the harmonic-contrast head but keep the harmonic views
    "phantom_no_harmonic": ["--no-harmonic"],
    # remove the differentiable transit decoder
    "phantom_no_decoder": ["--no-decoder"],
    # remove the harmonic views themselves
    "phantom_no_harmonic_views": [
        "--views", "global,local,odd,even,secondary,cent_global,cent_local"
    ],
    # remove the centroid channels
    "phantom_no_centroid": ["--views", "global,local,odd,even,secondary,half,double"],
    # the Shallue & Vanderburg input set, PHANTOM architecture
    "phantom_sv_views": ["--views", "sv"],
    # only light-curve-derived scalars: the configuration that transfers to TESS
    "phantom_derived_only": ["--scalar-groups", "derived"],
    # no scalar inputs at all
    "phantom_no_scalars": ["--scalar-groups", "none"],
    # no catalogue diagnostic statistics
    "phantom_no_dv": ["--scalar-groups", "transit,stellar,derived"],
}


def build_jobs(data: str, epochs: int) -> list[tuple[str, list[str]]]:
    base = [sys.executable, os.path.join(REPO, "scripts", "train.py"),
            "--data", data, "--epochs", str(epochs)]
    jobs = []

    for seed in HEADLINE_SEEDS:
        jobs.append((f"phantom_full/s{seed}", base + [
            "--model", "phantom", "--tag", "phantom_full", "--seed", str(seed)]))
        jobs.append((f"astronet/s{seed}", base + [
            "--model", "astronet", "--tag", "astronet", "--seed", str(seed),
            "--views", "sv", "--scalar-groups", "none"]))

    for tag, extra in ABLATIONS.items():
        for seed in ABLATION_SEEDS:
            jobs.append((f"{tag}/s{seed}", base + [
                "--model", "phantom", "--tag", tag, "--seed", str(seed)] + extra))

    # Second protocol: the TCE-level random split used by earlier published work.
    for seed in TCE_SPLIT_SEEDS:
        jobs.append((f"phantom_full_tce/s{seed}", base + [
            "--model", "phantom", "--tag", "phantom_full", "--seed", str(seed),
            "--split-mode", "tce"]))
        jobs.append((f"astronet_tce/s{seed}", base + [
            "--model", "astronet", "--tag", "astronet", "--seed", str(seed),
            "--split-mode", "tce", "--views", "sv", "--scalar-groups", "none"]))
    return jobs


def worker(gpu: str, q: "queue.Queue", log_dir: str, results: list, lock):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    while True:
        try:
            name, cmd = q.get_nowait()
        except queue.Empty:
            return
        safe = name.replace("/", "_")
        log_path = os.path.join(log_dir, f"{safe}.log")
        t0 = time.time()
        with open(log_path, "w") as fh:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60
        with lock:
            results.append((name, rc, dt))
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"[gpu{gpu}] {name}: {status} ({dt:.1f} min)  "
                  f"{len(results)} done, {q.qsize()} queued", flush=True)
        q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/dr24_views.h5")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--log-dir", default="logs/runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on job names")
    args = ap.parse_args()

    jobs = build_jobs(args.data, args.epochs)
    if args.only:
        jobs = [j for j in jobs if args.only in j[0]]
    print(f"{len(jobs)} jobs", flush=True)
    if args.dry_run:
        for name, cmd in jobs:
            print(f"  {name}: {' '.join(cmd[2:])}")
        return 0

    os.makedirs(args.log_dir, exist_ok=True)
    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)

    results, lock = [], threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(g, q, args.log_dir, results, lock), daemon=True)
        for g in args.gpus.split(",")
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [r for r in results if r[1] != 0]
    print(f"\n{len(results)} jobs in {(time.time()-t0)/60:.0f} min; {len(failed)} failed",
          flush=True)
    for name, rc, _ in failed:
        print(f"  FAILED {name} (rc={rc}) -- see {args.log_dir}/{name.replace('/','_')}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
