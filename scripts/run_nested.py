"""Nested ablation chain: successive deletions in a fixed order, five seeds each.

Unlike ``run_sweep.py``'s leave-one-out grid, every configuration here is a
strict subset of the one above it, so the cumulative cost of withdrawing
evidence is measured on a single chain rather than inferred across alternative
configurations. The last link shares AstroNet's input set exactly, which makes
the final step of the chain a controlled architecture-only contrast.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]

ALL_VIEWS = "global,local,odd,even,secondary,half,double,cent_global,cent_local"

# Ordered chain. Each entry removes evidence relative to its predecessor; the
# first four links already exist in results/runs from the leave-one-out sweep
# (full, no_dv, derived_only, no_scalars) and are not re-run here.
CHAIN = [
    ("nest_n4_no_centroid", ["--views", "global,local,odd,even,secondary,half,double",
                             "--scalar-groups", "none"]),
    ("nest_n5_no_harmonic", ["--views", "global,local,odd,even,secondary",
                             "--scalar-groups", "none"]),
    ("nest_n6_no_secondary", ["--views", "global,local,odd,even",
                              "--scalar-groups", "none"]),
    ("nest_n7_sv_only", ["--views", "sv", "--scalar-groups", "none"]),
]


def build_jobs(data: str, epochs: int) -> list[tuple[str, list[str]]]:
    base = [sys.executable, os.path.join(REPO, "scripts", "train.py"),
            "--data", data, "--epochs", str(epochs), "--model", "phantom"]
    return [
        (f"{tag}/s{seed}", base + ["--tag", tag, "--seed", str(seed)] + extra)
        for tag, extra in CHAIN
        for seed in SEEDS
    ]


def worker(gpu: str, q: "queue.Queue", log_dir: str, results: list, lock):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    while True:
        try:
            name, cmd = q.get_nowait()
        except queue.Empty:
            return
        log_path = os.path.join(log_dir, f"{name.replace('/', '_')}.log")
        t0 = time.time()
        with open(log_path, "w") as fh:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
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
    ap.add_argument("--run-dir", default="results/runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.data, args.epochs)
    if not args.force:
        pending = []
        for name, cmd in jobs:
            tag = cmd[cmd.index("--tag") + 1]
            seed = cmd[cmd.index("--seed") + 1]
            if not os.path.exists(
                os.path.join(args.run_dir, f"{tag}_seed{seed}_group.json")
            ):
                pending.append((name, cmd))
        if len(jobs) - len(pending):
            print(f"skipping {len(jobs) - len(pending)} already-completed runs",
                  flush=True)
        jobs = pending

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
    threads = [threading.Thread(target=worker,
                                args=(g, q, args.log_dir, results, lock), daemon=True)
               for g in args.gpus.split(",")]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [r for r in results if r[1] != 0]
    print(f"\n{len(results)} jobs in {(time.time()-t0)/60:.0f} min; "
          f"{len(failed)} failed", flush=True)
    for name, rc, _ in failed:
        print(f"  FAILED {name} (rc={rc})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
